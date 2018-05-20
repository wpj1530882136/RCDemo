#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import bottle
from bottle import route, run
import threading
import json
import numpy as np
import spacy
from time import sleep

'''
This file is taken and modified from R-Net by Minsangkim142
https://github.com/minsangkim142/R-net
'''
nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

app = bottle.Bottle()
query = []
response = ""

@app.get("/")
def home():
    with open('demo.html', 'r') as fl:
        html = fl.read()
        return html

@app.post('/answer')
def answer():
    passage = bottle.request.json['passage']
    question = bottle.request.json['question']
    print("received question: {}".format(question))
    # if not passage or not question:
    #     exit()
    global query, response
    query = (passage, question)
    while not response:
        sleep(0.1)
    print("received response: {}".format(response))
    response_ = {"answer": response}
    response = []
    return response_

class Demo(object):
    def __init__(self, model, config):
        run_event = threading.Event()
        run_event.set()
        threading.Thread(target=self.demo_backend, args = [model, config, run_event]).start()
        app.run(port=8080, host='0.0.0.0')
        try:
            while 1:
                sleep(.1)
        except KeyboardInterrupt:
            print("Closing server...")
            run_event.clear()

    def demo_backend(self, model, config, run_event):
        global query, response

        while run_event.is_set():
            sleep(0.1)
            if query:
                context = word_tokenize(query[0].replace("''", '" ').replace("``", '" '))
                question = word_tokenize(query[1].replace("''", '" ').replace("``", '" '))
                p_w_ids = [model.term_vocab.convert_to_ids(context)]
                q_w_ids = [model.term_vocab.convert_to_ids(question)]
                p_c_ids = [model.char_vocab.convert_to_ids(context, is_term=False)]
                q_c_ids = [model.char_vocab.convert_to_ids(question, is_term=False)]
                print("debug passage")
                print(p_w_ids)
                print("debug context")
                print(q_w_ids)
                print("debug pc")
                print(p_c_ids)
                print("debug qc")
                print(q_c_ids)
                p_len = len(p_w_ids[0])
                q_len = len(q_w_ids[0])
                for idx, ids in enumerate(p_c_ids[0]):
                    p_c_ids[0][idx] = (ids + [0] * (config.max_char_num - len(ids)))[:config.max_char_num]

                for idx, ids in enumerate(q_c_ids[0]):
                    q_c_ids[0][idx] = (ids + [0] * (config.max_char_num - len(ids)))[:config.max_char_num]

                print("debug pc")
                print(p_c_ids)
                print("debug qc")
                print(q_c_ids)

                p_chars = np.zeros((1, p_len, config.max_char_num)).astype('int32')
                p_chars[0][:, :] = p_c_ids[0][:p_len]

                q_chars = np.zeros((1, q_len, config.max_char_num)).astype('int32')
                q_chars[0][:, :] = q_c_ids[0][:q_len]

                start_id = [0]
                end_id = [0]
                p_lengths = [p_len]
                q_lengths = [q_len]

                feed_dict = {model.p: p_w_ids,
                             model.q: q_w_ids,
                             model.p_char: p_chars,
                             model.q_char: q_chars,
                             model.p_length: p_lengths,
                             model.q_length: q_lengths,
                             model.start_label: start_id,
                             model.end_label: end_id,
                             model.dropout_keep_prob: config.dropout_keep_prob}

                yp1, yp2 = model.sess.run([model.yp1, model.yp2], feed_dict = feed_dict)
                yp2[0] += 1
                response = "".join(context[yp1[0]:yp2[0]])
                query = []
