import tensorflow as tf

# 输入层
x = tf.constant([0.9, 0.85], shape=[1, 2])
# # 声明w1和w2两个变量作为weight参数
# w1 = tf.Variable(tf.constant([[0.2, 0.1, 0.3], [0.2, 0.4, 0.3]], shape=[2, 3]), name="w1")
# w2 = tf.Variable(tf.constant([0.2, 0.5, 0.25], shape=[3, 1]), name="w2")

# # b1和b2作为biase偏置项参数
# b1 = tf.constant([-0.3, 0.1, 0.2], shape=[1, 3], name="b1")
# b2 = tf.constant([-0.3], shape=[1], name="b2")

"""
在大多数实际情况下，我们并不能事先知道网络的参数。
对于weight权重，通常的做法是：将它初始化为一个随机矩阵。
对于biase偏置项参数，通常会用zeros()函数或者zeros()函数进行初始化。
"""
# 此处使用随机种子参数seed，这样可以保证每次运行得到的结果是一样的。
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name="w2")
# 将偏置项参数初始化
b1 = tf.Variable(tf.zeros([1, 3]))
b2 = tf.Variable(tf.ones(1))

# 初始化全部变量

init_op = tf.global_variables_initializer()

a = tf.matmul(x, w1) + b1
y = tf.matmul(a, w2) + b2

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y))
