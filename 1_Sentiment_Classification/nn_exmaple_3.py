from __future__ import print_function
import numpy as np
import tensorflow as tf

'''
Different to nn_example_2.py, this code shows how to organise all data-related code into a class.

The else of the code is the same as the last example.
'''

# Data simulation
class Data(object):
    def __init__(self, train_size, test_size, nb_feature, nb_classes, batch_size):
        self.train_size = train_size
        self.test_size = test_size
        self.nb_feature = nb_feature
        self.nb_classes = nb_classes
        self.batch_size = batch_size

        self.num_epoches = int(self.train_size / self.batch_size)

        self._generate_all_data()

        self.start = 0

    def dense_to_one_hot(self, dense_classes):
        """Convert class labels from scalars to one-hot vectors"""
        nb_samples, nb_classes = len(dense_classes), max(dense_classes) + 1
        labels_one_hot = []
        for i in dense_classes:
            line = np.zeros(nb_classes)
            line[i] = 1
            labels_one_hot.append(line)
        return np.array(labels_one_hot).reshape(len(dense_classes), max(dense_classes) + 1)

    def _generate_all_data(self):
        self.data_x = np.random.choice((0, 1), size=(self.train_size + self.test_size, self.nb_feature))
        sums = [np.sum(x) for x in self.data_x]
        classes = [s % nb_classes for s in sums]
        self.data_y = self.dense_to_one_hot(classes)

    def get_batch_train(self):
        x = self.data_x[self.start: self.start + self.batch_size]
        y = self.data_y[self.start:self.start + self.batch_size]
        self.start += self.batch_size
        return (x, y)

    def get_test(self):
        x = self.data_x[-self.test_size:]
        y = self.data_y[-self.test_size:]
        return (x, y)


def comput_acc(pred, target):
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy


# Parameters
learning_rate = 0.01
training_epochs = 10000
batch_size = 100
display_step = 50

# Network Parameters
n_hidden_1 = 8  # 1st layer number of features
n_hidden_2 = 8  # 2nd layer number of features
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

train_size, test_size, nb_feature, nb_classes = 1000, 100, 16, 3
data = Data(train_size, test_size, nb_feature, nb_classes, batch_size)
total_batch = data.num_epoches
x_test, y_test = data.get_test()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = data.get_batch_train()
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))
