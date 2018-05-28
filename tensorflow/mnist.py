"""
Simple NN MNIST classifier connected to TensorBoard.

tf.summary.<PANEL>("<NAME>", <TENSOR>, <?>)

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def preprocess():
    return input_data.read_data_sets()

def neural_network():
    """
    Initializes the network archictecture.
    """

    # Input - MNIST Images
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, 784], name="x-input") # input
        y_ = tf.placeholder(tf.int64, [None], name="y-target") # target

    # Input - Reshaped
    with tf.name_scope("input_reshape"):
        x_2d = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image("input", x_2d, 10)

    def weight_variable(shape):
        """
        Initialize a weight variable.
        """
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias_variable(shape):
        """
        Initialize a bias variable.
        """
        return tf.Variable(tf.constant(0.1), shape=shape)

    def nn_layer(layer_name, input_tensor, input_dim, output_dim,\
            activation_fn=tf.nn.relu):
        """
        Initialize an NN layer.
        """
        with tf.name_scope(layer_name):
            with tf.name_scope("weights"):
                weights = weight_variable([input_dim, output_dim])
            with tf.name_scope("biases"):
                weights = bias_variable([output_dim])
            with tf.name_scope("Wx_plus_b"):
                preactivation = tf.matmul(input_tensor, weights) + biases
            activation = act(preactivation, name="activation")
            tf.summary.histogram("activations", activation)
            return activation

    # Hidden Layers
    hidden1 = nn_layer("hidden1", x, 784, 500)
    hidden2 = nn_layer("hidden2", hidden1, 500, 500)

    # Dropout
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        dropped = tf.nn.dropout(hidden2, keep_prob)
    
    # Prediction
    y = nn_layer("y-prediction", dropped, 500, 10)

    # Loss Function
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=y_, logits=y)

    # Optimizer
    with tf.name_scope("optimizer"):
        train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(
            cross_entropy)

    # Accuracy
    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_predictions"):
            correct_predictions = tf.equal(tf.argmax(y, 1), y_)
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                tf.float32))

    # Merge summaries, and write out
    """
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    """
