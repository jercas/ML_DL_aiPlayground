# -*- coding: utf-8 -*-
#from keras.datasets import mnist
#(x_train, y_train),(x_test, y_test) = mnist.load_data()

import numpy as np
import copy
a=[[1,1,1],[2,2,2]]
a=np.reshape(a,(2,3))
print(a)
b=copy.deepcopy(a)
print(b)
print('after')
b=np.reshape(b,(3,2))
print(b)
print(a)