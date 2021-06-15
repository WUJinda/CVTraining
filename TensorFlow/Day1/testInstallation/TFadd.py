import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([3.0, 4.0], name="b")

result = a + b
# print(result)
# sess = tf.Session()
# sess.run(result)

#define session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print(sess.run(result))
    sess.close()
# if __name__ == "main":
#     sess = tf.Session()
#     sess.run(result)
