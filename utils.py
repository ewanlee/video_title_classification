import pickle
import codecs
from itertools import izip
import os
import tensorflow as tf
import sys
import numpy as np
import time
from tqdm import tqdm

def invert_dict_fast(dic):
    """
    invert dictionary

    Inputs:
    - dic (dict): the object dictionary

    Outputs:
    - NO_NAME (dict): the inverted dictionary
    """
    return dict(izip(dic.itervalues(), dic.iterkeys()))


def create_inverse_dict():
    with open('data/vocab.dic', 'rb') as f:
        vocab_dict = pickle.load(f)
    with open('data/inverse_vocab.dic', 'wb') as f:
        pickle.dump(invert_dict_fast(vocab_dict), f)


def index2word(unmatchs, invert_vocab_dict):
    """
    invert the indeies list to words list of the sentence

    Inputs:
    - sent (list): the word indeies in the vocabulary of the sentence
    - invert_vocab_dict (dict): the inverse dict

    Outputs:
    - unmatchs_words (list): the words of all sentences
    """
    unmatchs_words = []

    for unmatch in unmatchs:
        words = []
        for i in unmatch:
            try:
                word = invert_vocab_dict[i]
                words.append(word)
            except KeyError as e:
                pass
        unmatchs_words.append(' '.join(words))

    return unmatchs_words


def label2class(fname):
    """
    convert label to class
    """

    reload(sys)
    sys.setdefaultencoding("utf-8")

    print('convert label to class ...')
    label_class_dict = {}
    with codecs.open('data/categories.txt', 'r', 'utf-8') as f:
        for line in f.readlines():
            _label, _class = line.strip().split()
            label_class_dict[_label] = _class

    unmatches = []
    with codecs.open(fname, 'r', 'utf-8') as f:
        for line in f.readlines():
            temp_list = line.strip().split()
            temp_list[-2] = '[' + label_class_dict[temp_list[-2]] + ']'
            temp_list[-1] = '[' + label_class_dict[temp_list[-1]] + ']'
            unmatches.append(' '.join(temp_list))

    with codecs.open(fname, 'w', 'utf-8') as f:
        for u in unmatches:
            f.write(u + '\n')


def unmatched_sample(
        sess, model, X_val, y_val, batch_size,
        fname='data/textcnn_unmatchs.txt'):
    """
    get the unmatched sample info: (sentence, true label, pred label)

    Inputs:
    - sess (tf.Session): the active tensorflow session
    - model (textcnn, textrnn and so on): the prediction model
    - X_val (numpy.array): the word indeies of sentences
    - y_val (numpy.array): the true label of sentences
    - batch_size (int): the mini-batch size

    this method will create a file that recodes unmatched samples infomation
    """
    reload(sys)
    sys.setdefaultencoding("utf-8")
    create_inverse_dict()

    print('start to generate unmatched sample info ...')
    # remove previous generated file
    if os.path.exists(fname):
        os.remove(fname)

    number_examples = len(X_val)

    with open('data/label_index.map', 'rb') as f:
        classes_dict = pickle.load(f)
    invert_classes_dict = invert_dict_fast(classes_dict)

    with open('data/inverse_vocab.dic', 'rb') as f:
        invert_vocab_dict = pickle.load(f)

    for start, end in tqdm(zip(
            range(0, number_examples, batch_size),
            range(batch_size, number_examples, batch_size))):
        st = time.clock()
        feed_dict = {model.input_x: X_val[start:end],
                model.dropout_keep_prob: 1}

        predictions = sess.run([model.predictions], feed_dict)[0]
        unmatched_index = True ^ np.equal(
                list(map(int, predictions)), y_val[start:end])
        unmatchs = np.array(X_val[start:end])[unmatched_index]
        unmatches_words = index2word(unmatchs, invert_vocab_dict)

        pred_labels = [invert_classes_dict[pred] for pred in predictions[unmatched_index]]
        true_labels = [invert_classes_dict[label] for label in np.array(y_val[start:end])[unmatched_index]]

        for words, true_label, pred_label in zip(
                unmatches_words, true_labels, pred_labels):
            with codecs.open(fname, 'a', 'utf-8') as f:
                f.write('{}\t{}\t{}\n'.format(
                    words, true_label, pred_label))
        # convert label to class
    label2class(fname)
