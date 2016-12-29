import tensorflow as tf
import random
import pickle
import numpy as np
from collections import defaultdict

'''
Tweets classification using neural network
'''


class NN(object):
    def __init__(self, n_input_layer, n_hidden_layer_1, n_hidden_layer_2, n_hidden_layer_3, n_hidden_layer_4,
                 n_output_layer, learning_rate):
        self.n_input_layer = n_input_layer
        self.n_hidden_layer_1 = n_hidden_layer_1
        self.n_hidden_layer_2 = n_hidden_layer_2
        self.n_hidden_layer_3 = n_hidden_layer_3
        self.n_hidden_layer_4 = n_hidden_layer_4
        self.n_output_layer = n_output_layer
        self.learning_rate = learning_rate

        with tf.name_scope("input"):
            self.add_input()

        with tf.name_scope("hidden_1"):
            h1 = self.add_layer(self.xs, self.n_input_layer, self.n_hidden_layer_1)

        with tf.name_scope("hidden_2"):
            h2 = self.add_layer(h1, self.n_hidden_layer_1, self.n_hidden_layer_2)

        with tf.name_scope("hidden_3"):
            h3 = self.add_layer(h2, self.n_hidden_layer_2, self.n_hidden_layer_3)

        with tf.name_scope("hidden_4"):
            h4 = self.add_layer(h3, self.n_hidden_layer_3, self.n_hidden_layer_4)

        with tf.name_scope("out_put"):
            self.pred = self.add_layer(h4, self.n_hidden_layer_4, self.n_output_layer, activation=tf.nn.softmax)

        with tf.name_scope("cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.ys))
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def add_input(self):
        self.xs = tf.placeholder(tf.float32, [None, self.n_input_layer])
        self.ys = tf.placeholder(tf.float32, [None, self.n_output_layer])

    def add_layer(self, x, in_size, out_size, activation=None):
        W = tf.Variable(tf.random_normal([in_size, out_size]))
        b = tf.Variable(tf.zeros([out_size]) + 0.1)
        h = tf.matmul(x, W) + b
        h = tf.nn.relu(h) if activation is None else activation(h)
        return h


def get_batch_data(batch_size):
    # load training data
    filename = "training.csv"
    lines = open(filename, encoding="utf-8").readlines()

    # load vocabulary
    vocab = pickle.load(open('lexcion.p', 'rb'))
    vocab_index = defaultdict(int)
    for i in range(len(vocab)):
        vocab_index[vocab[i]] = i + 1

    while True:
        file_index = random.sample(range(len(lines)), batch_size)
        batch_file = [lines[i] for i in file_index]

        x_batch = np.zeros((batch_size, len(vocab) + 1))
        y_batch = []
        for i, line in enumerate(batch_file):
            y, x = line.split(":%:%:%:")
            words = x.strip().split(" ")
            for word in words:
                x_batch[i][vocab_index[word]] = 1

            y_batch.append(eval(y))

        yield (x_batch, y_batch)


# 几乎和training data 加载方式一样
def get_test_data():
    # load training data
    filename = "tesing.csv"
    lines = open(filename, encoding="utf-8").readlines()

    # load vocabulary
    vocab = pickle.load(open('lexcion.p', 'rb'))
    vocab_index = defaultdict(int)
    for i in range(len(vocab)):
        vocab_index[vocab[i]] = i + 1

    x_test = np.zeros((len(lines), len(vocab) + 1))
    y_test = []
    for i, line in enumerate(lines):
        y, x = line.split(":%:%:%:")
        words = x.strip().split(" ")
        for word in words:
            x_test[i][vocab_index[word]] = 1

        y_test.append(eval(y))

    return (x_test, y_test)


def comput_acc(pred, target):
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


def run():
    # model parameters
    n_input_layer = 34963 + 1
    n_hidden_layer_1 = 1024
    n_hidden_layer_2 = 1024
    n_hidden_layer_3 = 1024
    n_hidden_layer_4 = 1024
    n_output_layer = 3  # 3 classes

    # training parameters
    batch_size = 256
    epoch = 100000
    update_step = 32
    learning_rate = 0.01

    model = NN(n_input_layer, n_hidden_layer_1, n_hidden_layer_2, n_hidden_layer_3, n_hidden_layer_4, n_output_layer,
               learning_rate)

    data = get_batch_data(batch_size)

    x_test, y_test = get_test_data()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        pre_accuracy = 0
        for i in range(epoch):
            x_batch, y_batch = data.__next__()
            _, train_cost = sess.run([model.train_op, model.cost], feed_dict={model.xs: x_batch, model.ys: y_batch})

            if i % update_step == 0:
                test_cost = sess.run([model.cost], feed_dict={model.xs: x_test, model.ys: y_test})
                print("step: %s\n, train cost: %s, test cost: %s" % (i, train_cost, test_cost))

                acc = comput_acc(model.pred, model.ys)
                train_acc = sess.run(acc, feed_dict={model.xs: x_batch, model.ys: y_batch})
                test_acc = sess.run(acc, feed_dict={model.xs: x_test, model.ys: y_test})
                print("Accuracy\n, train acc: %s, test acc: %s" % (train_acc, test_acc))

                # save model
                if test_acc > pre_accuracy:
                    saver.save(sess, "./model/nn_model.ckpt")
                    print("Model saved at step %s" % i)
                    pre_accuracy = test_acc

                print("-" * 50)


if __name__ == '__main__':
    run()
