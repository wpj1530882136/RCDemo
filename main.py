#!/bin/bin/python
import tensorflow as tf
import os
from tensorflow.rc_model import model

def demo():
    graph = tf.Graph()
    with graph.as_default() as g:
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver()
            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))


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
def __main__():
    demo()
