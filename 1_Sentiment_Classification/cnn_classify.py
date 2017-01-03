import tensorflow as tf
import pickle
import numpy as np
from collections import defaultdict

'''
Tweets classification using neural network. Word embeddings are used is this example.

Result:

step: 20480
Train cost: 0.598435, test cost: 7.20083
Accuracy
Train acc: 0.78125, test acc: 0.586345
Model saved at step 20480
'''


class CNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, embedding_size, n_filters, filter_sizes, sequence_length, learning_rate, vocab_size, n_classes):
        self.embedding_size = embedding_size
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.n_classes = n_classes

        with tf.name_scope("input"):
            self.add_input()

        with tf.name_scope("embedding"):
            self.pred = self.add_embedding_layer()

        # CalculateMean cross-entropy loss
        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.ys))

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.cost)
            self.train_op = optimizer.apply_gradients(grads_and_vars)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.ys, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.accuracy = accuracy

    def add_input(self):
        self.xs = tf.placeholder(tf.int32, [None, self.sequence_length])
        self.ys = tf.placeholder(tf.float32, [None, self.n_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def add_embedding_layer(self):
        with tf.device("/cpu:0"):
            embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1, 1))
            embedded_chars = tf.nn.embedding_lookup(embeddings, self.xs)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # convolution + pooling layer
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv_maxpool_%s" % filter_size):
                filter_shape = [filter_size, self.embedding_size, 1, self.n_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
                conv = self._conv2d(embedded_chars_expanded, W)
                h = tf.nn.relu(tf.nn.bias_add(conv, b))

                # Maxpooling over the outputs
                ksize = [1, self.sequence_length - filter_size + 1, 1, 1]
                strides = [1, 1, 1, 1]
                pooled = self._max_pool(h, ksize, strides)

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.n_filters * len(self.filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # output
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_filters_total, self.n_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.n_classes]))
            output = tf.nn.xw_plus_b(h_drop, W, b)

        return output

    def _conv2d(self, x, W):
        # strides的定义：[1, x, y, 1]，x横轴方向上的step，y纵轴方向上的step
        # 支持SAME和Valid两种padding方式，same的feature map的维度和输入维度相同，valid不进行填充
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")

    def _max_pool(self, x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
        # ksize是pooling大小
        # strides是步长
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding="VALID")


def sent_2_idx(filename, text, max_sentence_length, vocab_size):
    # load vocabulary
    vocab = pickle.load(open(filename, 'rb'))
    print("Total vocabularies: ", len(vocab))
    print("Actual use %s words" % vocab_size)
    vocab = vocab[:vocab_size - 1]

    vocab_index = defaultdict(int)
    for i in range(len(vocab)):
        vocab_index[vocab[i]] = i + 1

    sent_idxs = []
    for line in text:
        sent_idx = [vocab_index[word] for word in line.split(" ")]
        sent_idx = sent_idx + [0] * (max_sentence_length - len(sent_idx))
        sent_idxs.append(sent_idx[:max_sentence_length])
    sent_idxs = np.array(sent_idxs).reshape(len(text), max_sentence_length)
    return sent_idxs


def load_data(filename, sequence_length, vocab_size):
    # load training data
    # filename = "training.csv"
    lines = open(filename, encoding="utf-8").readlines()

    y_text, x_text = [], []
    for line in lines:
        y, x_txt = line.split(":%:%:%:")
        x_txt.strip()
        x_text.append(x_txt)
        y_text.append(eval(y))
    y_text = np.array(y_text)

    document_length = [len(line.split(" ")) for line in lines]
    print("max document length: ", max(document_length))
    print("min document length: ", min(document_length))
    print("average document length: ", np.mean(document_length))
    print("Actual document length: ", sequence_length)

    x_text = sent_2_idx("lexcion.p", x_text, sequence_length, vocab_size)

    shuffle_indices = np.random.permutation(np.arange(len(y_text)))

    x_shuffled = x_text[shuffle_indices]
    y_shuffled = y_text[shuffle_indices]

    return (x_shuffled, y_shuffled)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def comput_acc(pred, target):
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


def run():
    # model parameters
    embedding_size = 128
    n_classes = 3  # 3 classes
    n_filters = 128
    filter_sizes = (3, 4, 5)
    sequence_length = 32
    vocab_size = 10000

    # training parameters
    batch_size = 128
    epoch = 5000
    display_step = 64
    learning_rate = 0.01
    dropout_keep_prob = 0.5

    model = CNN(embedding_size, n_filters, filter_sizes, sequence_length, learning_rate, vocab_size, n_classes)

    x_train, y_train = load_data("training.csv", sequence_length, vocab_size)
    x_test, y_test = load_data("tesing.csv", sequence_length, vocab_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        pre_accuracy = 0
        batches = batch_iter(list(zip(x_train, y_train)), batch_size, epoch)
        i = 0
        for batch in batches:
            i += 1
            x_batch, y_batch = zip(*batch)
            _, train_cost = sess.run([model.train_op, model.cost], feed_dict={model.xs: x_batch, model.ys: y_batch,
                                                                              model.dropout_keep_prob: dropout_keep_prob})

            if i % display_step == 0:
                test_cost = sess.run(model.cost,
                                     feed_dict={model.xs: x_test, model.ys: y_test, model.dropout_keep_prob: 1.0})
                print("step: %s\nTrain cost: %s, test cost: %s" % (i, train_cost, test_cost))

                acc = comput_acc(model.pred, model.ys)
                train_acc = sess.run(acc, feed_dict={model.xs: x_batch, model.ys: y_batch, model.dropout_keep_prob: 1.})
                test_acc = sess.run(acc, feed_dict={model.xs: x_test, model.ys: y_test, model.dropout_keep_prob: 1.})
                print("Accuracy\nTrain acc: %s, test acc: %s" % (train_acc, test_acc))

                # save model
                if test_acc > pre_accuracy:
                    saver.save(sess, "./model/cnn_model.ckpt")
                    print("Model saved at step %s" % i)
                    pre_accuracy = test_acc
                print("-" * 50)


if __name__ == '__main__':
    run()
