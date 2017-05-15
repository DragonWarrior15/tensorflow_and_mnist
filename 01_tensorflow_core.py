# the central unit of data in tensorflow is called tensor
# rank of a tensor is the no of dimensions in it
# tensorflow graph is created using computational nodes

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
# Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
# note that the nodes need to be evaluated in order to produce the values stored in them as the outputs

