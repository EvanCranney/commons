"""
Reference material for importing / exporting TensorFlow graphs from a .meta
file. Used for long-term storage and back-up of graphs.

Uses MNIST as the motivating example.
"""

#IS_SAVED = True

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data",
    one_hot=True)


# Preamble: TensorFlow Functions
def weight_variable(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias_variable(shape, name):
    return tf.Variable(tf.constant(0.1), shape, name=name)

# (1) Network Architecture
with tf.name_scope("input-layer"):
    x = tf.placeholder(tf.float32, [None,784], name="x-input")
    y_ = tf.placeholder(tf.float32, [None,10], name="y-target")

with tf.name_scope("nn-hidden-layer"):
    hidden1_W = weight_variable([784, 500], name="hidden_W")
    hidden1_b = bias_variable([500], name="hidden_b")
    hidden1_preact = tf.matmul(x, hidden1_W) + hidden1_b
    hidden1 = tf.nn.relu(hidden1_preact)

with tf.name_scope("output-layer"):
    output_W = weight_variable([500,10], name="output_W")
    output_b = bias_variable([10], name="output_b")
    y = tf.matmul(hidden1, output_W) + output_b

with tf.name_scope("loss-operation"):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

with tf.name_scope("train-operation"):
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.name_scope("accuracy-operation"):
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# (2) Train Network - First Time, not Saved
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # train
    for _ in range(20):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x:batch_xs, y_:batch_ys})
        print(sess.run(accuracy, feed_dict={x:mnist.test.images,
                                            y_:mnist.test.labels}))

    # save
    saver.save(sess, "./mnist_example")
    #saver.export_meta_graph("./mnist_example.meta")

# (3) Load Network - Second Time, Saved
with tf.Session() as sess:
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    print("ACCURACY BEFORE SAVE:")
    print(sess.run(accuracy, feed_dict={x:mnist.test.images,
                                        y_:mnist.test.labels}))

    saver.restore(sess, "./mnist_example")
    print("ACCURACY AFTER SAVE:")
    print(sess.run(accuracy, feed_dict={x:mnist.test.images,
                                        y_:mnist.test.labels}))
