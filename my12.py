# #coding=utf-8
# a=10
# while True:
#     b=3
#     a=a+b
#     print("step1")
#     print(a)
#     print("step2")

import tensorflow as tf

state = tf.Variable(0, name='counter')
print(state.name)

one = tf.constant(1)
# 常量加变量还是变量
new_value = tf.add(state, one)
# 把新的值给state变量
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()  # 初始化所有变量

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))