import tensorflow as tf
from RL.settings import RANDOM_SEED
tf.set_random_seed(RANDOM_SEED)
from net_2d import SimpleSoftMaxNet_2D
from RL.settings import run_settings
# a = tf.random_uniform([1])
# b = tf.random_normal([1])
weights = tf.get_variable('weights_last', [9, 2], initializer=tf.contrib.layers.xavier_initializer())

# Repeatedly running this block with the same graph will generate the same
# sequences of 'a' and 'b'.

net = SimpleSoftMaxNet_2D(run_settings())
init = tf.global_variables_initializer()
print("Session 1")
with tf.Session(graph=tf.get_default_graph()) as sess1:
  # print(sess1.run(a))  # generates 'A1'
  # print(sess1.run(a))  # generates 'A2'
  # print(sess1.run(b))  # generates 'B1'
  # print(sess1.run(b))  # generates 'B2'
  sess1.run(init)
  [weights_l] = sess1.run([net.tvars])
  #[weights_l] = sess1.run([weights])
  print weights_l

print("Session 2")
with tf.Session(graph=tf.get_default_graph()) as sess2:
  # print(sess2.run(a))  # generates 'A1'
  # print(sess2.run(a))  # generates 'A2'
  # print(sess2.run(b))  # generates 'B1'
  # print(sess2.run(b))  # generates 'B2'
  sess2.run(init)
  [weights_l] = sess2.run([net.tvars])
  #[weights_l] = sess2.run([weights])
  print weights_l