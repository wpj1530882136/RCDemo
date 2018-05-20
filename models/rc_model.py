# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""
from numpy.random import seed
seed(777)
from tensorflow import set_random_seed
set_random_seed(777)

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder
import layers.cnn_layer as cnn_layer
initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, term_vocab, char_vocab, args):

        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len
        self.max_char_num = args.max_char_num
        # the vocab
        self.term_vocab = term_vocab
        self.char_vocab = char_vocab
        self.emb_size = self.term_vocab.embed_dim

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode_back()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])

        self.p_mask = tf.cast(self.p, tf.bool)  # [batch, p_len]
        self.q_mask = tf.cast(self.q, tf.bool)  # [batch, q_len]

        self.p_char = tf.placeholder(tf.int32, [None, None, self.max_char_num], name='p_char')
        self.q_char = tf.placeholder(tf.int32, [None, None, self.max_char_num], name='q_char')

        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def _embed(self):

        """
        The embedding layer, question and passage share embeddings
        """
        with tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.term_vocab.size(), self.term_vocab.embed_dim),
                initializer=tf.constant_initializer(self.term_vocab.embeddings),
                trainable=True
            )
            self.p_word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)

        with tf.variable_scope('char_embedding'):
            self.char_embeddings = tf.get_variable(
                'char_embeddings',
                shape=(self.char_vocab.size(), self.char_vocab.embed_dim),
                initializer=tf.constant_initializer(self.char_vocab.embeddings),
                trainable=True
            )
            self.p_char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.p_char)  # [batch, seqlen, max_char_num, embedding_size]
            self.q_char_emb = tf.nn.embedding_lookup(self.char_embeddings, self.q_char)

            self.p_char_emb = self.cnn_emb(self.p_char_emb, "p_emb")
            self.q_char_emb = self.cnn_emb(self.q_char_emb, "q_emb")
            '''
            self.p_char_emb = tf.reshape(self.p_char_emb, [-1, self.max_char_num, self.emb_size])
            self.q_char_emb = tf.reshape(self.q_char_emb, [-1, self.max_char_num, self.emb_size])

            self.p_char_emb = cnn_layer.conv(self.p_char_emb, self.emb_size,
                                        bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=None)
            self.q_char_emb = cnn_layer.conv(self.q_char_emb, self.emb_size,
                                        bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=True)

            self.p_char_emb = tf.reduce_max(self.p_char_emb, axis=1)  # [batch*seqlen, 1, emb_size]
            self.q_char_emb = tf.reduce_max(self.q_char_emb, axis=1)

            batch_size = tf.shape(self.p_word_emb)[0]
            self.p_char_emb = tf.reshape(self.p_char_emb, [batch_size, -1, self.emb_size])
            self.q_char_emb = tf.reshape(self.q_char_emb, [batch_size, -1, self.emb_size])

            self.p_char_emb = tf.nn.dropout(self.p_char_emb, 0.95)
            self.q_char_emb = tf.nn.dropout(self.q_char_emb, 0.95)
            '''
        self.p_emb = tf.concat([self.p_word_emb, self.p_char_emb], -1)
        self.q_emb = tf.concat([self.q_word_emb, self.q_char_emb], -1)


#        self.p_emb = cnn_layer.highway(self.p_emb, size=self.hidden_size, scope="highway", dropout=0.1, reuse=None)
#        self.q_emb = cnn_layer.highway(self.q_emb, size=self.hidden_size, scope="highway", dropout=0.1, reuse=True)


    def _encode_back(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size)
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size)
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _encode(self):
        with tf.variable_scope("Embedding_Encoder_Layer"):
            self.sep_p_encodes = cnn_layer.residual_block(self.sep_p_encodes,
                                num_blocks=1,
                                num_conv_layers=4,
                                kernel_size=5,
                                mask=self.p_mask,
                                num_filters=self.hidden_size,
                                input_projection = True,
                                num_heads=1,
                                seq_len=self.p_length,
                                scope="Encoder_Residual_Block",
                                bias=False,
                                dropout=1-self.dropout_keep_prob)
            self.sep_q_encodes = cnn_layer.residual_block(self.sep_q_encodes,
                                num_blocks=1,
                                num_conv_layers=4,
                                kernel_size=5,
                                mask=self.q_mask,
                                num_filters=self.hidden_size,
                                input_projection=True,
                                num_heads=1,
                                seq_len=self.q_length,
                                scope="Encoder_Residual_Block",
                                reuse=True,  # Share the weights between passage and question
                                bias=False,
                                dropout=1-self.dropout_keep_prob)

    def _match_back(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p_length, self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _match(self):
        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            S = cnn_layer.optimized_trilinear_for_attention([self.sep_p_encodes, self.sep_q_encodes], input_keep_prob = self.dropout_keep_prob)
            mask_q = tf.expand_dims(self.q_mask, 1) #[batch, 1, q_len]
            S_ = tf.nn.softmax(cnn_layer.mask_logits(S, mask = mask_q))
            mask_c = tf.expand_dims(self.p_mask, 2) #[batch, p_len, 1]
            S_T = tf.transpose(tf.nn.softmax(cnn_layer.mask_logits(S, mask = mask_c), dim = 1),(0,2,1)) #[batch, q_len, p_len]
            self.c2q = tf.matmul(S_, self.sep_q_encodes) #[batch, p_len, emb_size]
            self.q2c = tf.matmul(tf.matmul(S_, S_T), self.sep_p_encodes) #[batch, p_len, q_len] * [batch, q_len, p_len] * [batch, p_len, emb_size]
            self.match_p_encodes = tf.concat(
                [self.sep_p_encodes,
                 self.c2q,
                 self.sep_p_encodes * self.c2q,
                 self.sep_p_encodes * self.q2c
                 ], -1)

    def _fuse_back(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion'):
            #self.fuse_p_encodes, _ = rnn('lstm', self.match_p_encodes, self.p_length,
             #                            self.hidden_size, layer_num=1)
            self.fuse_p_encodes = tc.layers.fully_connected(self.match_p_encodes, self.hidden_size)
            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

    def _fuse(self):
        with tf.variable_scope("Model_Encoder_Layer"):
            inputs = self.match_p_encodes
            self.enc = [cnn_layer.conv(inputs, self.hidden_size, name="input_projection")]
            for i in range(3):
                if i % 2 == 0:  # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], self.dropout_keep_prob)
                self.enc.append(
                    cnn_layer.residual_block(self.enc[i],
                                   num_blocks=1,
                                   num_conv_layers=2,
                                   kernel_size=5,
                                   mask=self.p_mask,
                                   num_filters=self.hidden_size,
                                   num_heads=1,
                                   seq_len=self.p_length,
                                   scope="Model_Encoder",
                                   bias=False,
                                   reuse=True if i > 0 else None,
                                   dropout=1-self.dropout_keep_prob)
                )

    def _decode_back(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes,
                [batch_size, -1, self.hidden_size]
            )
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], self.hidden_size]
            )[0:, 0, 0:, 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)

        outer = tf.matmul(tf.expand_dims(tf.nn.softmax(self.start_probs), axis=2),
                          tf.expand_dims(tf.nn.softmax(self.end_probs), axis=1))
        outer = tf.matrix_band_part(outer, 0, -1)
        self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
        self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
    def _decode(self):

        with tf.variable_scope("Output_Layer"):
            batch_size = tf.shape(self.start_label)[0]
            self.enc[1] = tf.reshape(self.enc[1], [batch_size, -1, self.hidden_size])
            self.enc[2] = tf.reshape(self.enc[2], [batch_size, -1, self.hidden_size])
            self.enc[3] = tf.reshape(self.enc[3], [batch_size, -1, self.hidden_size])
            p_mask = tf.reshape(self.p_mask, [batch_size, -1])
            start_logits = tf.squeeze(
                cnn_layer.conv(tf.concat([self.enc[1], self.enc[2]], axis=-1), 1, bias=False, name="start_pointer"),
                -1)  # [batch, p_len]
            end_logits = tf.squeeze(
                cnn_layer.conv(tf.concat([self.enc[1], self.enc[3]], axis=-1), 1, bias=False, name="end_pointer"),
                -1)  # [batch, p_len]
            self.logits = [cnn_layer.mask_logits(start_logits, mask=p_mask),
                           cnn_layer.mask_logits(end_logits, mask=p_mask)]

            self.start_probs, self.end_probs = [l for l in self.logits]
            self.start_probs = tf.nn.softmax(self.start_probs, -1)
            self.end_probs = tf.nn.softmax(self.end_probs, -1)
            
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(self.start_probs), axis=2),
                              tf.expand_dims(tf.nn.softmax(self.end_probs), axis=1))
            outer = tf.matrix_band_part(outer, 0, -1)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

    def _compute_loss(self):
        """
        The loss function
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            #self.check(batch)
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_char: batch['passage_char_ids'],
                         self.q_char: batch['question_char_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: dropout_keep_prob}
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.term_vocab.get_id(self.term_vocab.pad_token)
        max_bleu_4 = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix)
                        max_bleu_4 = bleu_rouge['Bleu-4']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_char: batch['passage_char_ids'],
                         self.q_char: batch['question_char_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: 1.0}
            start_probs, end_probs, loss = self.sess.run([self.start_probs,
                                                          self.end_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': sample['answers'],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
    '''
    def highway(self, x, size=None, activation=None,
                num_layers=2, scope="highway", dropout_keep_prob=1.0, reuse=None):
        with tf.variable_scope(scope, reuse):
            if size is None:
                size = x.shape.as_list()[-1]
            else:
                x = self.conv(x, size, name="input_projection", reuse=reuse)
            for i in range(num_layers):
                T = self.conv(x, size, bias=True, activation=tf.sigmoid,
                         name="gate_%d" % i, reuse=reuse)
                H = self.conv(x, size, bias=True, activation=activation,
                         name="activation_%d" % i, reuse=reuse)
                H = tf.nn.dropout(H, dropout_keep_prob)
                x = H * T + x * (1.0 - T)
            return x
    def conv(self, inputs, output_size, bias=None, activation=None, kernel_size=1, name="conv", reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            shapes = inputs.shape.as_list()
            if len(shapes) > 4:
                raise NotImplementedError
            elif len(shapes) == 4:
                filter_shape = [1, kernel_size, shapes[-1], output_size]
                bias_shape = [1, 1, 1, output_size]
                strides = [1, 1, 1, 1]
            else:
                filter_shape = [kernel_size, shapes[-1], output_size]
                bias_shape = [1, 1, output_size]
                strides = 1
            conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
            kernel_ = tf.get_variable("kernel_",
                                      filter_shape,
                                      dtype=tf.float32,
                                      regularizer=regularizer,
                                      initializer=initializer_relu() if activation is not None else initializer())
            w_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            outputs = conv_func(inputs, w_filter, strides, "VALID")
            if bias:
                outputs += tf.get_variable("bias_",
                                           bias_shape,
                                           regularizer=regularizer,
                                           initializer=tf.zeros_initializer())
            if activation is not None:
                return activation(outputs)
            else:
                return outputs
    '''
    def cnn_emb(self, inputs, scope):
        # inputs shape=[batch, seqlen, max_char_num, embedding_size]

        with tf.variable_scope(scope):
            batch_size = tf.shape(inputs)[0]
            seqlen = tf.shape(inputs)[1]
            cnn_inputs = tf.reshape(inputs, [batch_size, -1, self.char_vocab.embed_dim])  # [batch, seqlen *max_char_num, embedding_size]
            filter_shape = [3, self.char_vocab.embed_dim , self.char_vocab.embed_dim]
            w_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            cnn_inputs = tf.nn.conv1d(cnn_inputs, w_filter, stride=1, padding="SAME")  # [batch, seqlen*max_char_num, embedding_size]
            cnn_inputs = tf.expand_dims(cnn_inputs, 1)  # [batch, 1, seqlen*max_char_num, embedding_size]
            cnn_emb = tf.nn.max_pool(cnn_inputs,
                                     ksize=[1, 1, self.max_char_num, 1],
                                     strides=[1, 1, self.max_char_num, 1],
                                     padding="VALID")
            cnn_emb = tf.reshape(cnn_emb, [batch_size,seqlen, self.char_vocab.embed_dim])
            cnn_emb_dropout = tf.nn.dropout(cnn_emb, self.dropout_keep_prob)
        return cnn_emb_dropout

    def check(self, batch):
        self.logger.info('------------question_token-------------------')
        q_tokens = batch['question_token_ids']
        for i in range(len(q_tokens)):
            lst = self.term_vocab.recover_from_ids(q_tokens[i])
            for k in range(len(lst)):
                self.logger.info(lst[k])

        self.logger.info('------------passage_token-------------------')
        p_tokens = batch['passage_token_ids']
        for i in range(len(p_tokens)):
            lst = self.term_vocab.recover_from_ids(p_tokens[i])
            for k in range(len(lst)):
                self.logger.info(lst[k])
        
        self.logger.info('------------question_char-------------------')
        q_chars = batch['question_char_ids'].tolist()
        print(q_chars)
        for i in range(len(q_chars)):
            for j in range(len(q_chars[i])):
                lst = self.char_vocab.recover_from_ids(q_chars[i][j])
                for k in range(len(lst)):
                    self.logger.info(lst[k])

        self.logger.info('------------passage_char-------------------')
        p_chars = batch['passage_char_ids'].tolist()
        for i in range(len(p_chars)):
            for j in range(len(p_chars[i])):
                lst = self.char_vocab.recover_from_ids(p_chars[i][j])
                for k in range(len(lst)):
                    self.logger.info(lst[k])
