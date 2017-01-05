from data_helper import batch_iter
import tensorflow as tf

'''
Use Char_RNN to generate Chinese Tang Poetry.
'''


class RNN(object):
    def __init__(self, batch_size, maxlen, num_units, num_rnn_layers, vocab_size, model="lstm"):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.model = model
        self.num_units = num_units
        self.num_rnn_layers = num_rnn_layers
        self.vocab_size = vocab_size

        self.build_model()

    def input(self):
        self.xs = tf.placeholder(tf.int32, [self.batch_size, self.maxlen])
        self.ys = tf.placeholder(tf.int32, [self.batch_size, self.maxlen])

    def embedding_layer(self):
        # embedding look up only use CPU
        with tf.device("/cpu:0"):
            embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.num_units], -1, 1))
            embedded_chars = tf.nn.embedding_lookup(embeddings, self.xs)
            self.embedded_chars = tf.reshape(embedded_chars, [self.batch_size, self.maxlen,
                                                              self.num_units])  # output dim: [128 batch, 28 steps, 128 hidden]

    def cell(self):
        if self.model == "lstm":
            cell_fun = tf.nn.rnn_cell.BasicLSTMCell
        elif self.model == "gru":
            cell_fun = tf.nn.rnn_cell.GRUCell
        elif self.model == "rnn":
            cell_fun = tf.nn.rnn_cell.BasicRNNCell
        else:
            raise Exception("Model error")

        cell = cell_fun(self.num_units, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_rnn_layers, state_is_tuple=True)
        _init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(cell, self.embedded_chars, initial_state=_init_state)
        outputs = tf.reshape(outputs, [-1, self.num_units])
        self.cell_outputs = outputs

    def output(self):
        W = tf.Variable(tf.random_uniform([self.num_units, self.vocab_size], -1, 1))
        b = tf.Variable(tf.constant(0.1, shape=[self.vocab_size]))
        logits = tf.nn.xw_plus_b(self.cell_outputs, W, b)
        self.logits = tf.nn.softmax(logits)

    def loss(self):
        targets = tf.reshape(self.ys, [-1])
        weights = tf.ones_like(targets, dtype=tf.float32)
        loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits], [targets], [weights], self.vocab_size)
        self.loss = tf.reduce_mean(loss)

    def train(self):
        self.learning_rate = tf.Variable(0.0, trainable=False)
        t_vars = tf.trainable_variables()  # Returns all variables created with `trainable=True`.
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, t_vars), 5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, t_vars))

    def build_model(self):
        with tf.name_scope("input"):
            self.input()

        with tf.name_scope("embedding"):
            self.embedding_layer()

        with tf.name_scope("cell"):
            self.cell()

        with tf.name_scope("softmax"):
            self.output()

        with tf.name_scope("loss"):
            self.loss()

        with tf.name_scope("train"):
            self.train()


def run():
    # training parameters
    batch_size = 128
    num_epochs = 100
    maxlen = 16
    step = 3
    next_n = 1

    # model parameters
    num_units = 128
    num_rnn_layers = 2
    vocab_size = 10000

    model = RNN(batch_size, maxlen, num_units, num_rnn_layers, vocab_size)

    texts = batch_iter("poetry.txt", batch_size, num_epochs, maxlen, vocab_size, step, next_n)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())

        i = 0
        for text in texts:
            x_batch, y_batch = zip(*text)
            sess.run(tf.assign(model.learning_rate, 0.5 * (0.99 ** i)))

            train_loss, _ = sess.run([model.loss, model.train_op],
                                     feed_dict={model.xs: x_batch, model.ys: y_batch})
            print("Epoch: %s, loss: %s" % (i, train_loss))
            i += 1
        if i % 100 == 0:
            saver.save(sess, 'model/rnn.ckpt', global_step=i)


if __name__ == '__main__':
    run()
