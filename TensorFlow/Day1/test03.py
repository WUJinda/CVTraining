import tensorflow as tf

with tf.variable_scope("one"):
    a = tf.get_variable("a", [1], initializer=tf.constant_initializer(1.0))
with tf.variable_scope("one", reuse=True):
    a2 = tf.get_variable("a", [1])
    print(a.name, a2.name)
"""
 tf.variable_scope()方法创建变量空间时，reuse默认False，所以创建新变量，
 此变量拥有新的name属性，如果退出此空间对应的with结构，在相同空间下还以默认方法
 创建变量则会报错，因为此空间中已经存在name属性为a的变量；在不同空间下则不会报错。
"""
with tf.variable_scope("two", reuse=False):
    v1 = tf.get_variable("a", [1])
    print(v1.name)


