# taken from
# https://www.tensorflow.org/get_started/get_started

# the central unit of data in tensorflow is called tensor
# rank of a tensor is the no of dimensions in it
# tensorflow graph is created using computational nodes

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
# note that the nodes need to be evaluated in order to produce the values stored in them as the outputs

# To actually evaluate the nodes, we must run the computational graph within a session.
# A session encapsulates the control and state of the TensorFlow runtime

sess = tf.Session()
print(sess.run([node1, node2]))
# [3.0, 4.0]

# can combine the nodes through operators
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))
# node3:  Tensor("Add:0", shape=(), dtype=float32)
# sess.run(node3):  7.0

# here we are completely dealing with constants..
# we can use variables/placeholders also
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

# we can pass a dictionary of inputs to the above, of any dimensions
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
# 7.5
# [ 3.  7.]

# we can make the operation slightly more complex
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))
print(sess.run(add_and_triple, {a: [3, 4], b:[4.5, 8]}))
# 22.5
# [ 22.5  36. ]

# all the above placeholders are constants and cannot be modified on the go
# for the purposes of training a model
# variables give us this freedom they are constructed with a type and initial value
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# constants are initialized when tf.constant is called, avariables don't hold this property
# they need to be initialized with a initializer call
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x:[1,2,3,4]}))
# [ 0.          0.30000001  0.60000002  0.90000004]

# after building the model, we would also like to be able to convey a loss function for the same
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
# reduce_sum reduces the vector sum of squares to a scalar equivalent
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
# 23.66

# we would like to be able to change the values of W and b in order
# to tune our model. variables initialiized using tf.Variable() can be
# modified through tf.assign
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
# 0.0

