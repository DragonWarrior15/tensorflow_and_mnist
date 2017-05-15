import os
import tensorflow as tf

input_data_path = '../MNIST_data/'

# special wrapper to read the mnis data
from tensorflow.examples.tutorials.mnist import input_data
# here we are using a one hot encoded vector so as to have uniform "brightness"
# across the pixels
mnist = input_data.read_data_sets(input_data_path, one_hot = True)

# train and test images are stored in the objects
# mnist.train.images and mnist.test.images respectively

# print the shapes of the different sets, note that the data set is a numpy array itself
# and the shape parameter can be used here
print (mnist.train.images.shape)
# (55000, 784)
print (mnist.test.images.shape)
# (10000, 784)

# heavy mathematical computations are usually done outside of python.. such
# as in numpy, the backend caculations are implemented in C
# the overhead cost of this going in and out of python can be quite high
# in GPUs and distributed environments
# tensorflow overcomes this by defining everything in a graph like manner
# where all computations can be done outside of python independently

x = tf.placeholder(tf.float32, [None, 784])
# here x is a placeholder for any "matrix" of second dimension 784
# the first dimension is None, meaning the no of examples in our set
# can be as many as we want

# add weights and the bias term
# here we are not passing any value to them, meaning we are initializing them
# all to arrays/matrices filled with zeros using tf.zeros
# usual syntax is W = tf.Variable([.3], tf.float32)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax function to get probability values
y = tf.nn.softmax(tf.matmul(x, W) + b)

# actual y values
y_ = tf.placeholder(tf.float32, [None, 10])

# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

# training function and optimizer
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# initialize sessions and variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) # reset values to incorrect defaults.

# loop to run all the steps of the decent
for _ in range(10000):
    # feed the data in batches
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})

is_correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# the above give a boolean array as output, cast it to float
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# 0.923
