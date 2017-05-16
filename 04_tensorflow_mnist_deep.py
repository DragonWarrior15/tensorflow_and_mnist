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

# we are going to build a multilayer network.. hence functions to initialize them
# we are going to use rectified linear units as neurons in our model
# hence it is advisable to get normally distributed neurons to break the symmetry
# and also to prevent zero gradiens/dead neurons
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# model parameters
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
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
