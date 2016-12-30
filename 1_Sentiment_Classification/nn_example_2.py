'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf


def get_batch_data(batch_size, total_batch):
    while True:
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            line = np.random.choice([0, 1], size=(16))
            x_batch.append(line)
            sum = np.sum(line)
            if sum % 3 == 0:
                y = [1, 0, 0]
            elif sum % 3 == 1:
                y = [0, 1, 0]
            else:  # 2
                y = [0, 0, 1]
            y_batch.append(y)
        x_batch = np.array(x_batch).reshape(batch_size, 16)
        y_batch = np.array(y_batch).reshape(batch_size, 3)
        yield (x_batch, y_batch)


# 几乎和training data 加载方式一样
def get_test_data():
    # load test data
    x_test = []
    y_test = []
    for i in range(100):
        line = np.random.choice([0, 1], size=(16))
        x_test.append(line)
        sum = np.sum(line)
        if sum % 3 == 0:
            y = [1, 0, 0]
        elif sum % 3 == 1:
            y = [0, 1, 0]
        else:  # 2
            y = [0, 0, 1]
        y_test.append(y)
    x_test = np.array(x_test).reshape(batch_size, 16)
    y_test = np.array(y_test).reshape(batch_size, 3)
    return (x_test, y_test)


def comput_acc(pred, target):
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 5

# Network Parameters
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_input = 16  # MNIST data input (img shape: 28*28)
n_classes = 3  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

total_batch = 100
data = get_batch_data(batch_size, total_batch)
x_test, y_test = get_test_data()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = data.__next__()
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            print("Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: batch_x, y: batch_y}))
