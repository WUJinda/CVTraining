import tensorflow as tf

# 理解变量空间的嵌套
a = tf.get_variable("a", [1], initializer=tf.constant_initializer(1.0))
print(a.name)
"""
    输出a:0
    "a"是此变量的名称
    ":0"表示是生成这个变量运算的第一个结果。
"""
with tf.variable_scope("one"):
    a2 = tf.get_variable("a", [1], initializer=tf.constant_initializer(1.0))
    print(a2.name)
"""
    输出one/a:0
    "one/"表示a所属的变量空间为one
    "a"是此变量的名称
    ":0"表示是生成这个变量运算的第一个结果。
"""
with tf.variable_scope("one"):
    with tf.variable_scope("two"):
        a4 = tf.get_variable("a", [1])
        print(a4.name)
    b = tf.get_variable("b", [1])
    print(b.name)
    # 因为退出了变量空间two

with tf.variable_scope("", reuse=True):
    a5 = tf.get_variable("one/two/a", [1])
    print(a5 == a4)
    # 可以直接通过带变量空间名称前缀的变量名来获取相应的变量。