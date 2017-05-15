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

