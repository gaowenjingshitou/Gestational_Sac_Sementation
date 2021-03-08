#coding==utf-8
#测试调试、断点

import tensorflow as tf
# print("++++++++start2++++++++++")

def aa():
    print("run in aa() start")
    print("step1")
    num1=3

    num2=num1*4
    num3=num2*5
    print("Step2")

    print("run in aa() end!!!")


if __name__=="__main__":
    print("main:step1")
    aa()
    print("main:step2")
    print("main:end!!!！")

import torch

print(torch.version.cuda)

import tensorflow as tf
print(tf.test.is_gpu_available())
import tensorflow as tf
print(tf.__version__)
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))