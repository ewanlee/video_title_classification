import jieba
import re
from ordered_set import OrderedSet
from collections import OrderedDict
from tqdm import tqdm
import pickle
import os

def load_data():
    """
    load sentences and corresponding labels from data file
    """
    sentences = []
    labels = []
    extract_label = ['0', '150000','122000','139000','125000','129000','115000','141000','190000']
    print('load data ...')
    with open("data/all_video_info.txt", 'r') as f:
        for line in tqdm(f.readlines()):
            temp_list = line.decode('utf-8').strip().split('\t')
            if temp_list[2] not in extract_label:
                sentences.append(temp_list[-1])
                labels.append(temp_list[2])

    return sentences, labels


def remove_emoji(sentences):
    """
    remove emoji

    Inputs:
    - sentences (list): the source sentences.

    Outputs:
    - NO_NAME (list): the sentences that emojis are removed.
    """
    print('remove emoji ...')
    return [re.sub('\[.*?\]', '', sent) for sent in tqdm(sentences)]


def cut(sentences):
    """
    cut the sentences with jieba

    Inputs:
    - sentences (list): the list of all sentences that will be cutted.

    Outputs:
    - cutted_sentences (list): the list of all sentences that be cutted.
    """
    print('cut sentences ...')
    cutted_sentences = [jieba.lcut(sent) for sent in tqdm(sentences)]
    return cutted_sentences


def remove_stop_words(cutted_sentences):
    """
    remove stop words from cutted sentences

    Inputs:
    - cutted_sentences (list): the list of all sentences that be cutted.

    Outputs:
    - cleaned_sentences (list): the list of all cutted sentences that removed stop words.
    """
    stop_words = OrderedSet()
    stop_words.add(' ')
    print('stop words load ...')
    with open('data/stopword.dic', 'r') as f:
        for line in tqdm(f.readlines()):
            stop_words.add(line.decode('utf-8').strip())

    print('remove stop words ...')
    cleaned_sentences = [list(OrderedSet(cutted_sent) - stop_words) for cutted_sent in tqdm(cutted_sentences)]
    return cleaned_sentences


def length_distribution(cleaned_sentences):
    """
    view the distribution of sentences length

    Inputs:
    - cleaned_sentences (list): the list of all cutted sentences that removed stop words.

    Outputs:
    - distri (dict): the dict that contains length distribution

    """
    print('calculate sentences length ...')
    lengths = [len(sent) for sent in tqdm(cleaned_sentences)]
    keys = list(set(lengths))
    keys.sort()
    distri = {}
    print('length distribution build ...')
    for i in tqdm(keys):
        distri[i] = lengths.count(i)
    return distri


def vocab_build(cleaned_sentences, saved_fn='data/vocab.dic'):
    """
    build the vocabulary list

    Inputs:
    - cleaned_sentences (list): each element is the words of sentence
    - saved_fn (string): the file name of saved vocabulary dictionary

    Outputs:
    - vocab (dict): (key, value) is (word, index)
    """
    if os.path.exists(saved_fn):
        with open(saved_fn, 'rb') as f:
            return pickle.load(f)

    vocab = OrderedDict()
    print('vocabulary words build ...')
    temp_list = []
    for words in tqdm(cleaned_sentences):
        temp_list.extend(words)
    vocab_set = OrderedSet(temp_list)
    print("vocabulary words' index build ...")
    for word in tqdm(list(vocab_set)):
        if word not in vocab.keys():
            vocab[word] = len(vocab)

    with open(saved_fn, 'wb') as f:
        pickle.dump(vocab, f)

    return vocab

def word2index(cleaned_sentences, vocab):
    """
    model for sentences.
    that is, every word in a sentence will transform to the index of this word
    in vocabulary dictionary.

    Inputs:
    - cleaned_sentences (list): each element is the words of sentence
    - vocab (dict): (key, value) is (word, index)

    Outputs:
    - modeled_sentences (list): each element is the index of words of sentence
    """
    modeled_sentences = []
    for sent in tqdm(cleaned_sentences):
        modeled_sentences.append([vocab[word] for word in sent])
    return modeled_sentences



if __name__ == '__main__':
    sentences, labels = load_data()
    cutted_sentences = cut(remove_emoji(sentences))
    cleaned_sentences = remove_stop_words(cutted_sentences)

    print('start build vocabulary list ...')
    vocab = vocab_build(cleaned_sentences)

    print('vocabulary list size: {}'.format(len(vocab)))

    print('modeling sentences ...')
    modeled_sentences = word2index(cleaned_sentences, vocab)
    print('the first 10 modeled sentences:')
    print(modeled_sentences[:10])

    with open('data/X.data', 'wb') as f:
        pickle.dump(modeled_sentences, f)
    with open('data/y.data', 'wb') as f:
        pickle.dump(labels, f)
