# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 11:28:45 2017

@author: jercas
"""
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation

def binaryClassification():
    '''For a single-input model with 2 classes (binary classification)'''
    # 构造Sequential序贯模型
    # 向model传递一个layer的list
    model = Sequential([
            Dense(32, input_dim=100),
            Activation('relu'),
            Dense(1, activation='sigmoid')
            ])

    ''' 
    通过.add()方法，逐个加入
    model = Sequential()
    model.add(Dense(32, input_shape=(784,)))
    model.add(Activation('relu'))
    '''
    
    # 指定输入数据shape
    # 第一层接受一个关于输入数据shape的参数，后面的各个层则可以自动的推导出中间数据的shape
    '''
    传递一个input_shape的关键字参数给第一层。
    input_shape是一个tuple类型的数据，其中也可以填入None，如果填入None则表示此位置可能是任何正整数。
    数据的batch大小不应包含在其中。
    model.add(Dense(32, input_shape=784))
    
    有些2D层，如Dense，支持通过指定其输入维度input_dim来隐含的指定输入数据shape。
    一些3D的时域层支持通过参数input_dim和input_length来指定输入shape。
    model.add(Dense(32, input_dim=784))
    
    如果需要为输入指定一个固定大小的batch_size（常用于stateful RNN网络）。
    可以传递batch_size参数到一个层中。
    例如想指定输入张量的batch大小是32，数据shape是（6，8），则需要传递batch_size=32和input_shape=(6,8)。
    '''
    
    # 编译
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    
    # 创建虚拟训练集
    data = np.random.random((1000,100))
    labels = np.random.randint(2, size=(1000,1))
    
    # 训练模型，在32个batch中进行10轮迭代
    model.fit(data, labels, epochs=10, batch_size=32)


def categoricalClassification():
    '''For a single-input model with 10 classes (categorical classification)'''
    # 创建模型
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(10, activation='softmax'))
    
    # 编译
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 创建数据集
    data = np.random.random((1000,100))
    labels = np.random.randint(10,size=(1000,1))
    
    # 将标签转换为独热编码分类
    one_hot_labels = keras.utils.to_categorical(labels,num_classes=10)
    
    # 训练模型，在32个batch中进行10轮迭代
    model.fit(data, one_hot_labels, epochs=10, batch_size=32)
    
    
binaryClassification()
#categoricalClassification()