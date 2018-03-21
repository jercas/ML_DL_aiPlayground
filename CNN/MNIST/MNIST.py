from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Flatten,Dense,Dropout
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import time, datetime, os
import matplotlib.pyplot as plt

#加载数据
(X_train,y_train),(X_test,y_test) = mnist.load_data()
#数据形态
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)
#打印数据
#plt.imshow(X_train[1])
log_dir = './log_dir'
os.mkdir(log_dir)

es = EarlyStopping(monitor='val_acc', patience=20)
mc = ModelCheckpoint(log_dir + '\\CIFAR10-EP-{epoch:02d}-ACC-{val_acc:.4f}.h5',
                     monitor='val_acc', save_best_only=True)
tb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False, embeddings_freq=0,
                 embeddings_layer_names=None, embeddings_metadata=None)

#数据预处理，1为通道数
X_train = X_train.reshape(X_train.shape[0], 28, 28 ,1)
X_test = X_test.reshape(X_test.shape[0], 28 ,28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#对标签进行预处理,每个样本维度为10
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#定义模型架构
model = Sequential() #初始化模型
#添加卷积层，32,3,3 通道数32、3*3的kenerl
model.add(Conv2D(32,(3,3),input_shape=(28,28,1)))
#添加激活层
model.add(Activation('relu'))
#添加卷积层，不需要在输入input_shape
model.add(Conv2D(32,(3,3)))
#添加激活层
model.add(Activation('relu'))
#添加池化层
model.add(MaxPooling2D(pool_size=(2,2),strides = 2))
#添加全连接层
model.add(Flatten()) #展开层,将图片像素展开
model.add(Dense(128,activation='relu'))#全连接层，128个神经元
#dropout层
model.add(Dropout(0.5))
#添加全连接层
model.add(Dense(10,activation='softmax'))

#编译，误差损失：categorical_crossentropy，优化方法：adam， 评估方法：accuracy
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

#训练
history = model.fit(X_train,
             y_train,
            batch_size=32,
            epochs=15,
            verbose=1,
            validation_split=0.2,
		    callbacks = [es, mc, tb])

#评估
score = model.evaluate(X_test,y_test, verbose=1)
print(score[0],score[1])
model.save("./MNIST_MODEL_{0}.h5".format(score[1]))

acc, loss = history.history['acc'], history.history['loss']
val_acc, val_loss = history.history['val_acc'], history.history['val_loss']
epoch = len(acc)
#summarize history for accuracy
plt.figure(figsize=(17, 5))
plt.subplot(121)
plt.plot(range(epoch), acc, label='Train')
plt.plot(range(epoch), val_acc, label='Test')
plt.title("Model Accuracy over " + str(epoch) + ' Epochs', size=15)
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend()
plt.grid(True)
plt.savefig("PerformanceAccuracy_"+str(score[1])+".png")

#summarize history for loss
plt.figure(figsize=(17, 5))
plt.subplot(121)
plt.plot(range(epoch), loss, label='Train')
plt.plot(range(epoch), val_loss, label='Test')
plt.title("Model Loss over " + str(epoch) + ' Epochs', size=15)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.grid(True)
plt.savefig("PerformanceLoss_"+str(score[1])+".png")
plt.show()