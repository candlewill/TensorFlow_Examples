import tensorflow as tf
import numpy as np

'''
Linear regression
'''


class Model(object):
    def __init__(self, learning_rate, input_size, output_size):
        self.xs = tf.placeholder(tf.float32, [None, input_size])
        self.ys = tf.placeholder(tf.float32, [None, output_size])

        self.W = W = tf.Variable(tf.random_normal([input_size, output_size]))
        self.b = b = tf.Variable(tf.zeros(output_size) + 0.1)

        self.pred = tf.matmul(self.xs, W) + b

        self.cost = tf.reduce_mean(tf.pow(self.pred - self.ys, 2)) / 2

        self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)


def run():
    learning_rate = 0.05
    input_size = 1
    output_size = 1
    model = Model(learning_rate, input_size, output_size)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        train_x = np.arange(100, step=0.1).reshape(-1, 1) / 10
        train_y = train_x * 20 + 10

        for epoch in range(1000):
            _, cost = sess.run([model.train_op, model.cost],
                               feed_dict={model.xs: train_x.reshape(-1, 1), model.ys: train_y.reshape(-1, 1)})

            print(cost)

            print(sess.run([model.W, model.b]))


if __name__ == '__main__':
    run()
