"""
Reference Example of Recurrent Neural Network (LSTM).
https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
https://www.tensorflow.org/versions/r1.1/tutorials/recurrent
https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537
"""
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

TIME_STEPS = 28         # length of unrolled time sequence, rows
N_INPUT = 28            # length of input sequence, cols
NUM_UNITS = 128         # hidden LSTM units

LEARNING_RATE = 0.002
NUM_LABELS = 10         # number of target labels
BATCH_SIZE = 128        # size of training batch


with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=[None,TIME_STEPS, N_INPUT])
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_LABELS])

with tf.name_scope("input-transformed"):
    x_unstack = tf.unstack(x, TIME_STEPS, 1)
    # convert from [batch_size, time_steps, n_input] to
    # ... [time_steps, batch_size, n_input]

with tf.name_scope("lstm"):
    lstm = tf.contrib.rnn.BasicLSTMCell(NUM_UNITS, forget_bias=1)
    lstm_out, _ = tf.contrib.rnn.static_rnn(lstm, x_unstack, dtype=tf.float32)

with tf.name_scope("output"):
    out_W = tf.Variable(tf.random_normal([NUM_UNITS, NUM_LABELS]))
    out_b = tf.Variable(tf.random_normal([NUM_LABELS]))
    y = tf.matmul(lstm_out[-1], out_W) + out_b

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=y, labels=y_))

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
        loss)

with tf.name_scope("accuracy"):
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epoch = 1
    while epoch < 800:
        batch_x, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
        batch_x = batch_x.reshape((BATCH_SIZE, TIME_STEPS, N_INPUT))
        sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y})

        if epoch % 10 == 0:
            score = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y})
            print("Accuracy: ", score)

        epoch += 1
