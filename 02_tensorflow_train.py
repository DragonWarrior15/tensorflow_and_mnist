# after completing 01_tensorflow_core.py, we jump into the tf.train API
# for training our machine learning models

# tf has inbuilt optimizers to slowly change the values of parameters
# to "learn"

import tensorflow as tf

# initialize parameters to incorrect defaults
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

optimizer = tf.train.GradientDescentOptimizer(0.01)
# here a single run of the optimizer just runs a single gradient decent step
# use the for loop to run several such iterations and get the opimal parameters set
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init) # reset values to incorrect defaults.
for i in range(100):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
    print(i, sess.run([W, b]))

print(sess.run([W, b]))
# [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]