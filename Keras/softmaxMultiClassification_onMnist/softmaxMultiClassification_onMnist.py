# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:26:36 2017

@author: jercas
"""

import numpy as np
# for reproducibility
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

def dataPreprocessing():
    """
        get data from data set, then doing data pre-
        Args:
        Returns:
            X_train: train data
            y_train: train label
            X_test:  test data
            y_test:  test label
    """
    #  X_train.shape = (60000, 28, 28)  y_train.shape = (60000, 1)
    # X_test.shape = (10000, 28, 28)   y_test.shape = (10000, 1)
    # all tpye is uint8's ndarray
    (X_train, y_train) , (X_test, y_test) = mnist.load_data()
    
    # preprocess
    # normalize
    # X_train reshape = (60000, 784)
    X_train = X_train.reshape(X_train.shape[0], -1) / 255
    # X_test reshape = (10000, 784)
    X_test = X_test.reshape(X_test.shape[0], -1) / 255
    
    # one_hot transform
    # y_train transform to one_hot coding , the shape of it is (60000, 10). ten classification
    y_train = np_utils.to_categorical(y_train, 10)
    # y_test transform to one_hot coding , the shape of it is (10000, 10). ten classification
    y_test = np_utils.to_categorical(y_test, 10)
    
    return X_train, y_train, X_test, y_test


def builder(X, y):
    """
        training until bulid a optimum classificate model
            Args:
            X: training data
            y: training label
        Returns:
            model: trained learning model
    """
    # get sequential model
    model = Sequential()
    # input layer
    model.add(Dense(32, input_dim=784, activation='relu'))
    # output layer
    model.add(Dense(10, activation='softmax'))
    
    # custom RMS optimizer
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # complie  model
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # start training model
    model.fit(X, y , epochs=10, batch_size=32)
    
    return model
    
def predictor(X, y, model):
    """
        test predict accuracy of trained learning model
        Args:
            X: testing data
            y: testing label
            model: trained learning model
        Returns:
            loss: model's predict loss
            accuracy: model's predict accuracy
    """
    loss, accuracy = model.evaluate(X, y)
    return loss, accuracy


def main():
    X_train, y_train, X_test, y_test = dataPreprocessing()
    model = builder(X_train, y_train)
    loss, accuracy = predictor(X_test, y_test, model)
    
    print('test loss:{0}'.format(loss))
    print('test accuracy:{0}'.format(accuracy))    
    
    
if __name__ == '__main__':
    main()