import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load the data
mnist = input_data.read_data_sets('data/raw/mnist/', one_hot=True)

# parameters
batch_size = 128
training_steps = 10001
learning_rate = 0.0008
n_hidden_units_1 = 32
keep_prob = 0.75
print_step = 100

# create the graph
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    kp = tf.placeholder(tf.float32)

    weights = {
        'w1': tf.Variable(tf.truncated_normal([784, n_hidden_units_1])),
        'out': tf.Variable(tf.truncated_normal([n_hidden_units_1, 10]))
    }
    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_units_1])),
        'out': tf.Variable(tf.zeros([10]))
    }

    # model
    nn = tf.matmul(x, weights['w1']) + biases['b1']
    nn = tf.nn.relu(nn)
    nn = tf.nn.dropout(nn, keep_prob=kp)
    logits = tf.matmul(nn, weights['out']) + biases['out']

    # optimizer
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                labels=y))
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # metric
    correct_preds = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))


# run the training steps
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    for ii in range(training_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        feed_dict = {x: batch_x, y: batch_y, kp: keep_prob}
        _, c = sess.run([optimize, cost], feed_dict=feed_dict)

        if ii % print_step == 0:
            feed_dict = {x: mnist.validation.images,
                         y: mnist.validation.labels, kp: 1.0}
            acc = sess.run(accuracy, feed_dict=feed_dict)
            print('[{:>5}] Cost: {:>10.6f}   Acc: {:>6.4f}'.format(ii, c, acc))

    feed_dict = {x: mnist.test.images, y: mnist.test.labels, kp: 1.0}
    print('Test accuracy: {:.4f}'.format(sess.run(accuracy,
                                         feed_dict=feed_dict)))
