#coding=utf-8
import tensorflow as tf


IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

#创建会话
sess = tf.Session()

#定义输入变量
xs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])


# 计算weight
def weigth_variable(shape):
    # stddev : 正态分布的标准差
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断正态分布
    return tf.Variable(initial)

# 计算biases
def bias_varibale(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



# 定义模型保存对象
saver = tf.train.Saver([W, b])








