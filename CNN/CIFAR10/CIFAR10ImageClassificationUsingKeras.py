
# coding: utf-8

# CIFAR10 Image Classification using Keras

# # Preprocess

# In[1]:


from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, pickle
import os
import datetime

def preprocess(nb_classes):
	"""
		training data preprocessing
		Args:
			nb_classes: classify categories
		Returns:
			X_train: (50000, 32, 32, 3)
			y_train: (10000, 32, 32, 3)
			X_test: (50000,)
			y_test: (10000,)
	"""
	# read data, X-32*32*3, y-1; training-50000, test-100000.
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()

	# Reshape: transform data shape from n*1 matrix to for input.
	y_train = y_train.reshape(y_train.shape[0])
	y_test = y_test.reshape(y_test.shape[0])

	# label transform from concrete value(0,1,2,3...10) to one-hot code([1,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0]...[0,0,0,0,0,0,0,0,0,0,1])
	# for softmax activation classify.
	y_train = to_categorical(y_train, nb_classes)
	y_test = to_categorical(y_test, nb_classes)

	print('X_train shape:', X_train.shape)
	print('y_train shape:', y_train.shape)
	print(X_train.shape[0], 'training samples')
	print(X_test.shape[0], 'validation samples')

	# Normalize: turn the range of value from [0,255] to [0,1]
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	return X_train, X_test, y_train, y_test


def modelBuild(nb_classes):
	"""
		模型采用类似于 VGG16 的结构：

		使用固定尺寸的小卷积核 (3x3)
		以2的幂次递增的卷积核数量 (64, 128, 256)
		两层卷积搭配一层池化
		全连接层没有采用 VGG16 庞大的三层结构，避免运算量过大，仅使用 128 个节点的单个FC
		权重初始化采用He Normal

		Args:
			nb_classes: classify categories
		Returns:
			model: compiled model architecture
	"""
	# sequential network architecture
	model = Sequential()
	# using a networks architecture like VGG16, but more smaller for a light weight compute
	x = Input(shape=(32, 32, 3))
	# static kernel size - 3*3
	# weights initialize - He Normal
	y = x
	# two time convolution match one pooling
	y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
					  kernel_initializer='he_normal')(y)
	y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
					  kernel_initializer='he_normal')(y)
	y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

	y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
					  kernel_initializer='he_normal')(y)
	y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
					  kernel_initializer='he_normal')(y)
	y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

	y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
					  kernel_initializer='he_normal')(y)
	y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
					  kernel_initializer='he_normal')(y)
	y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

	y = Flatten()(y)
	# drop out 50%
	y = Dropout(0.5)(y)
	# output layer: 10 units
	y = Dense(units=nb_classes, activation='softmax', kernel_initializer='he_normal')(y)

	model = Model(inputs=x, outputs=y, name='model1')
	# bulid & compiling & model architecture
	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
	model.summary()
	return model


def train(X_train, X_test, y_train, y_test, model, epoch, batch, data_augmentation=True):
	"""
		train learning model
		Args:
			X_train: (50000, 32, 32, 3)
			y_train: (10000, 32, 32, 3)
			X_test: (50000,)
			y_test: (10000,)
			model: complied model architecture
			epoch: training epoch
			batch: batch size
			data_augmentation: bool,whether to using image augment
		Returns:
			hypo: the object contains data of training process
			model: trained learning model
	"""
	start = time.time()
	log_dir = datetime.datetime.now().strftime('model_%Y%m%d_%H%M')
	os.mkdir(log_dir)

	es = EarlyStopping(monitor='val_acc', patience=20)
	mc = ModelCheckpoint(log_dir + '\\CIFAR10-EP{epoch:02d}-ACC{val_acc:.4f}.h5',
						 monitor='val_acc', save_best_only=True)
	tb = TensorBoard(log_dir=log_dir, histogram_freq=1,  write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

	if data_augmentation:
		aug = ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=True)
		aug.fit(X_train)
		gen = aug.flow(X_train, y_train, batch_size=batch)
		hypo = model.fit_generator(generator=gen,
								steps_per_epoch=50000/batch,
								epochs=epoch,
								validation_data=(X_test, y_test),
								callbacks=[es, mc, tb])
	else:
		start = time.time()
		hypo = model.fit(x=X_train,
					  y=y_train,
					  batch_size=batch,
					  epochs=epoch,
					  validation_data=(X_test, y_test),
					  callbacks=[es, mc, tb])

	print('\n@ Total Time Spent: %.2f seconds' % (time.time() - start))

	acc, val_acc = hypo.history['acc'], hypo.history['val_acc']
	m_acc, m_val_acc = np.argmax(acc), np.argmax(val_acc)

	print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
	print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
	return hypo, model


def plot_acc_loss(hypo):
	"""
		visualize result
		Args:
			hypo: trained learning model
			epoch: training epoch
		Returns:

	"""
	acc, loss = hypo.history['acc'], hypo.history['loss']
	val_acc, val_loss = hypo.history['val_acc'], hypo.history['val_loss']
	epoch = len(acc)
	plt.figure(figsize=(17, 5))
	plt.subplot(121)
	plt.plot(range(epoch), acc, label='Train')
	plt.plot(range(epoch), val_acc, label='Test')
	plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
	plt.legend()
	plt.grid(True)

	plt.subplot(122)
	plt.plot(range(epoch), loss, label='Train')
	plt.plot(range(epoch), val_loss, label='Test')
	plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
	plt.legend()
	plt.grid(True)
	plt.show()


def visualization(X_test, y_test, class_name, model):
	"""
	"""
	rand_id = np.random.choice(range(10000), size=10)
	X_pred = np.array([X_test[i] for i in rand_id])
	y_true = [y_test[i] for i in rand_id]
	y_true = np.argmax(y_true, axis=1)
	y_true = [class_name[name] for name in y_true]
	y_pred = model.predict(X_pred)
	y_pred = np.argmax(y_pred, axis=1)
	y_pred = [class_name[name] for name in y_pred]
	plt.figure(figsize=(15, 7))
	for i in range(10):
		plt.subplot(2, 5, i + 1)
		plt.imshow(X_pred[i].reshape(32, 32, 3), cmap='gray')
		plt.title('True: %s \n Pred: %s' % (y_true[i], y_pred[i]), size=15)
	plt.show()

def main():
	"""
		Cifar10 contains about 60000 pictures which can be distinguished into 10 categories. like below shows.
		Each picture is a 32*32 pixels color image.
	"""
	# Import data
	nb_classes = 10
	class_name = {
		0: 'airplane',
		1: 'automobile',
		2: 'bird',
		3: 'cat',
		4: 'deer',
		5: 'dog',
		6: 'frog',
		7: 'horse',
		8: 'ship',
		9: 'truck'}
	X_train, X_test, y_train, y_test = preprocess(nb_classes=nb_classes)

	model = modelBuild(nb_classes=nb_classes)
	hypo,model = train(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model=model,
				 epoch=100, batch=256, data_augmentation=True)

	loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
	print("Training Accuracy = %.2f %%     loss = %f" % (accuracy * 100, loss))
	loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
	print("Testing Accuracy = %.2f %%    loss = %f" % (accuracy * 100, loss))
	model.save('CIFAR10_model_with_data_augmentation.h5')

	plot_acc_loss(hypo)
	visualization(X_test=X_test, y_test=y_test, class_name=class_name, model=model)

if __name__ == "__main__":
	main()



