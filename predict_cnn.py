import tensorflow as tf
import os
import pickle
from tqdm import tqdm
from utils import invert_dict_fast
import codecs
from tflearn.data_utils import pad_sequences

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("ckpt_dir", "text_cnn_title_desc_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_string("unlabeled_data_location", "data/X_unlabeled.data", "unlabeled data location")
tf.app.flags.DEFINE_integer("batch_size", 64, "mini-batch size")
tf.app.flags.DEFINE_string("label_index_map_location", "data/label_index.map", "label to index map file location")
tf.app.flags.DEFINE_string("output_location", "data/output.label", "prediction file location")
tf.app.flags.DEFINE_integer('sentence_len', 15, 'sentence length')


def convert_lebel_to_class(fname):
    print('conver label to class ...')
    label_class_dict = {}
    with codecs.open('data/categories.txt', 'r', 'utf-8') as f:
        for line in f.readlines():
            _label, _class = line.strip().split()
            label_class_dict[_label] = _class

    class_list = []
    with codecs.open(fname, 'r', 'utf-8') as f:
        for line in f.readlines():
            class_list.append(label_class_dict[line.strip()])

    with codecs.open(fname, 'w', 'utf-8') as f:
        for cls in class_list:
            f.write(cls + '\n')


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    if os.path.exists(FLAGS.output_location):
        os.remove(FLAGS.output_location)

    with tf.Session(config=config) as sess:
        saver = None
        for filename in os.listdir(FLAGS.ckpt_dir):
            if 'meta' in filename:
                saver = tf.train.import_meta_graph(FLAGS.ckpt_dir + filename)
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name('input_x:0')
        keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')

        with open(FLAGS.unlabeled_data_location, 'rb') as f:
            X_unlabeled = pickle.load(f)
        predictions = graph.get_tensor_by_name('predictions:0')

        number_examples = len(X_unlabeled)
        batch_size = FLAGS.batch_size

        with open(FLAGS.label_index_map_location, 'rb') as f:
            class_dict = pickle.load(f)
        inverse_class_dict = invert_dict_fast(class_dict)

        with open('data/vocab.dic', 'rb') as f:
            vocab = pickle.load(f)
        vocab_size = len(vocab) + 1

        for start, end in tqdm(zip(range(0, number_examples, batch_size),
            range(batch_size, number_examples, batch_size))):

            X_pred = pad_sequences(X_unlabeled[start:end], maxlen=FLAGS.sentence_len, value=float(vocab_size - 1))
            feed_dict = {input_x: X_pred, keep_prob: 1}

            pred_labels = sess.run(predictions, feed_dict)

            pred_class = [inverse_class_dict[pred] for pred in pred_labels]
            with codecs.open(FLAGS.output_location, 'a', 'utf-8') as f:
                for cls in pred_class:
                    f.write(cls + '\n')

        convert_lebel_to_class(FLAGS.output_location)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    tf.app.run()
