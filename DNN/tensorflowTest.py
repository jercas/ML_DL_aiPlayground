# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 17:02:51 2017

@author: jercas
"""
import numpy as np 
import tensorflow as tf 
def test_tf_session(): 
    """ this method playing with tensorflow 'Session', 使用tensorlfow，你首先要创建一个图，然后通过会话来流动这张图，从而 生成对应的tensor，也就是一个个的矩阵 :return: """ 
    matrix1 = tf.constant([[1, 2, 3], [2, 3, 4]]) 
    matrix2 = tf.constant([[3, 4, 2], [1, 3, 4], [3, 4, 5]]) 
    product = tf.matmul(matrix1, matrix2) 
    sess = tf.Session() 
    result = sess.run(product) 
    print(result) 


def test_tf_variable(): 
    """ 这个方法示例'Variable'的作用，它相当于一个存储器，存储中间变量 :return: """ 
    # 首先我们定义一个Variable，名字叫state，初始值是38 
    state = tf.Variable(38, name='state') 
    add_value = tf.constant(3) 
    new_value = tf.add(state, add_value) 
    update = tf.assign(state, new_value) 
    # 使用variable，在用会话启动它之前要初始化一下'Variable' 
    # 要不然tf怎么知道你设定的初始值是多少呢？ 
    init_op = tf.initialize_all_variables() 
    
    with tf.Session() as sess: 
        sess.run(init_op) 
        sess.run(new_value) 
        for _ in range(3): 
            #执行这一步把state和add_value相加的值，得加到state自身 
            sess.run(update) 
            # 每一步执行之后我们看看state的值 
            print(sess.run(state)) 
          
            
def test_tf_feed_data(): 
    """ Feed data进入图之中，入口是placeholder，相当于占位符先把入口霸占一下， 等数据来了再从这里进入图之中 :return: """ 
    x = tf.placeholder(tf.float32, shape=(2, 3), name='matrix1') 
    y = tf.placeholder(tf.float32, shape=(3, 4), name='matrix2') 
    product = tf.matmul(x, y) 
    data_x = [[1, 2, 3], [3, 4, 2]] 
    data_y = [[2, 3, 4, 2], [1, 3, 4, 2], [2, 3, 4, 5]] 
    # 我们指定了两个数据流入的入口，并且固定了形状，如果输入不对会报错， 
    # 像这样正确的姿势塞进去，我们就能够得到product这个op的值 
    with tf.Session() as sess: 
        result = sess.run(product, feed_dict={x: data_x, y: data_y}) 
        print(result) 
        
if __name__ == '__main__': 
    test_tf_session()
    test_tf_variable() 
    test_tf_feed_data()
