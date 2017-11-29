# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tflearn.data_utils import pad_sequences, to_categorical
from tqdm import tqdm
from textcnn import TextCNN
from gensim.models import KeyedVectors
import os
import numpy as np

# configuration
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("num_classes", 33, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir", "text_cnn_title_desc_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 15, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 64, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 30, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters", 256, "number of filters")
tf.app.flags.DEFINE_boolean("multi_label_flag", False, "use multi label or single label.")
filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def label2index(labels):
    """
    convert meaning label to index label

    Inputs:
    - labels (list): the meaning label of all sentences

    Outputs:
    - classes (list): the class label of all sentences
    """
    classes_dict = {}
    print('we have {} labels, convert label to index ...'.format(len(set(labels))))
    for label in tqdm(set(labels)):
        if label not in classes_dict.keys():
            classes_dict[label] = len(classes_dict)
    classes = [classes_dict[label] for label in labels]
    return classes


def train_test_loader():
    """
    load data file and split to train set and validation set

    Outputs:
    - X_train, y_train, X_val, y_val (numpy.array)
    - n_classes (int): the classes number
    """
    with open('data/X.data', 'rb') as f:
        X = pickle.load(f)
    with open('data/y.data', 'rb') as f:
        y = label2index(pickle.load(f))

    n_classes = max(y) + 1
    X_train, y_train, X_val, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_val, y_val, n_classes


def main(_):
    X_train, X_val, y_train, y_val, n_classes = train_test_loader()
    with open('data/vocab.dic', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab) + 1
    print('size of vocabulary: {}'.format(vocab_size))

    # padding sentences
    X_train = pad_sequences(X_train, maxlen=FLAGS.sentence_len, value=float(vocab_size - 1))
    X_val = pad_sequences(X_val, maxlen=FLAGS.sentence_len, value=float(vocab_size - 1))
    # convert label to one-hot encode
    # to_categorical(y_train, n_classes)
    # to_categorical(y_val, n_classes)

    # create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        textcnn = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.num_classes,
                FLAGS.learning_rate, FLAGS.batch_size,
                FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len,
                vocab_size, FLAGS.embed_size, FLAGS.is_training, multi_label_flag=False)
        # Initialize save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + 'checkpoint'):
            print('restoring variables from checkpoint')
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:
                assign_pretrained_word_embedding(sess, vocab, vocab_size, textcnn)
        curr_epoch = sess.run(textcnn.epoch_step)

        # feed data and training
        number_of_training_data = len(X_train)
        batch_size = FLAGS.batch_size
        best_val_acc = 0.0
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, acc, counter = .0, .0, 0
            for start, end in zip(
                    range(0, number_of_training_data, batch_size),
                    range(batch_size, number_of_training_data, batch_size)):
                if epoch == 0 or counter == 0:
                    pass
                    # print('X_train[start:end]: {}'.format(X_train[start:end]))
                feed_dict = {
                        textcnn.input_x: X_train[start:end], textcnn.dropout_keep_prob: 0.5}
                if not FLAGS.multi_label_flag:
                    feed_dict[textcnn.input_y] = y_train[start:end]
                else:
                    feed_dict[textcnn.input_y_multilabel] = y_train[start:end]
                curr_loss, curr_acc, _ = sess.run(
                        [textcnn.loss_val, textcnn.accuracy, textcnn.train_op], feed_dict)
                loss, counter, acc = loss + curr_loss, counter + 1, acc + curr_acc

                if counter % 50 == 0:
                    print('Epoch {}\tBatch {}\tTrain Loss {}\tTrain Accuracy {}'.format(
                        epoch, counter, loss / float(counter), acc / float(counter)))
            print('going to increment epoch counter ...')
            sess.run(textcnn.epoch_increment)

            # validation
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, textcnn, X_val, y_val, batch_size)
                print("Epoch {} Validation Loss: {}\tValidation Accuracy: {}".format(
                    epoch, eval_loss, eval_acc))
                if eval_acc > best_val_acc:
                    best_val_acc = eval_acc
                    # save model to checkpoint
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    saver.save(sess, save_path, global_step=epoch)
                else:
                    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

        # report result
        test_loss, test_acc = do_eval(sess, textcnn, X_val, y_val, batch_size)

def assign_pretrained_word_embedding(
        sess, vocab, vocab_size, textCNN,
        word2vec_model_path='data/news_12g_baidubaike_20g_novel_90g_embedding_64.model'):
    print("using pre-trained word emebedding start ...")
    word2vec_model = KeyedVectors.load(word2vec_model_path)
    # create an empty word_embedding list
    word_embedding_2dlist = [[]] * vocab_size
    # assign empty for last word: 'PAD'
    word_embedding_2dlist[-1] = np.zeros(FLAGS.embed_size)
    # bound for random variables
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)

    count_exist = 0
    count_not_exist = 0

    # loop for each word
    for i, word in enumerate(list(vocab.keys())):
        word = word.encode('utf8')
        embedding = None
        try:
            embedding = word2vec_model[word]
        except Exception as e:
            embedding = None
        if embedding is not None:
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1
        else:
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1

    # convert to 2d array
    word_embedding_final = np.array(word_embedding_2dlist)
    # convert to tensor
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)
    # assign this value to our embeding variables of our model
    t_assign_embedding = tf.assign(textCNN.Embedding, word_embedding)
    sess.run(t_assign_embedding)

    print("word exists embedding: {}, word not exists embedding: {}".format(count_exist, count_not_exist))
    print("using pre-trained word emebedding end ...")


def do_eval(sess, textcnn, X_val, y_val, batch_size):
    number_examples = len(X_val)
    eval_loss, eval_acc, eval_counter = .0, .0, 0
    for start, end in zip(
            range(0, number_examples, batch_size),
            range(batch_size, number_examples, batch_size)):
        feed_dict = {textcnn.input_x: X_val[start:end], textcnn.dropout_keep_prob: 1}
        if not FLAGS.multi_label_flag:
            feed_dict[textcnn.input_y] = y_val[start:end]
        else:
            feed_dict[textcnn.input_y_multilabel] = y_val[start:end]
        curr_eval_loss, logits, curr_eval_acc = sess.run(
                [textcnn.loss_val, textcnn.logits, textcnn.accuracy], feed_dict)
        eval_loss, eval_acc, eval_counter = \
                eval_loss + curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1
    return eval_loss / float(eval_counter), eval_acc / float(eval_counter)


if  __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    tf.app.run()
