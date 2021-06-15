import tensorflow as tf

# define constant as input, and x is a mat(1*2)
x = tf.constant([[1.0, 2.0]])

# define variable as param
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# do matrix multiplication
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

init_p = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_p)
    print(sess.run(y))
    print(y)
