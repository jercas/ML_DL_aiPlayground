# import the modules we need
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras import metrics
from keras.optimizers import SGD,RMSprop,Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
import os, time, datetime
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.09
set_session(tf.Session(config=config))

#define the Sequential model
class CNNNet:

	@staticmethod
	def createNet(input_shapes,nb_class):

		feature_layers = [

		BatchNormalization(input_shape=input_shapes),
		Conv2D(64,(3,3),padding="same", kernel_initializer='he_normal'),
		Activation("relu"),
		BatchNormalization(),
		Conv2D(64,(3,3),padding="same", kernel_initializer='he_normal'),
		Activation("relu"),
		MaxPooling2D(pool_size=(2,2),strides=(2,2)),

		BatchNormalization(),
		Conv2D(128,(3,3),padding="same", kernel_initializer='he_normal'),
		Activation("relu"),
		BatchNormalization(),
		Dropout(0.5),
		Conv2D(128,(3,3),padding="same", kernel_initializer='he_normal'),
		Activation("relu"),
		MaxPooling2D(pool_size=(2,2),strides=(2,2)),

		BatchNormalization(),
		Dropout(0.5),
		Conv2D(256,(3,3),padding="same", kernel_initializer='he_normal'),
		Activation("relu"),
		Dropout(0.5),
		Conv2D(256,(3,3),padding="same", kernel_initializer='he_normal'),
		Activation("relu"),
		MaxPooling2D(pool_size=(2,2),strides=(2,2)),
		BatchNormalization()

		]

		classification_layer=[
		Flatten(),
		#Dense(512),
		#Activation("relu"),
		Dropout(0.5),
		Dense(units=nb_class, kernel_initializer='he_normal'),
		Activation("softmax")
		]

		model = Sequential(feature_layers+classification_layer)
		return model

#parameters
NB_EPOCH = 40
BATCH_SIZE = 128
VERBOSE = 1
VALIDATION_SPLIT = 0.2
IMG_ROWS=32
IMG_COLS = 32
NB_CLASSES = 10
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
INPUT_SHAPE =(IMG_ROWS,IMG_COLS,3)

#load cifar-10 dataset
(X_train,Y_train),(X_test,Y_test) = cifar10.load_data()

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

#X_train = X_train.reshape(X_train.shape[0],IMG_ROWS,IMG_COLS,3)
#X_test = X_test.reshape(X_test.shape[0],IMG_ROWS,IMG_COLS,3)

X_train /= 255
X_test /= 255

print(X_train.shape,"train shape")
print(Y_test.shape,"test shape")

#convert class vectors to binary class matrices
Y_train = to_categorical(Y_train,NB_CLASSES)
Y_test = to_categorical(Y_test,NB_CLASSES)

# init the optimizer and model
model = CNNNet.createNet(input_shapes=(32,32,3),nb_class=NB_CLASSES)
model.compile(loss="categorical_crossentropy",optimizer='adadelta',metrics=['accuracy'])
model.summary()

start = time.time()
log_dir = datetime.datetime.now().strftime('model_%Y%m%d_%H%M')
os.mkdir(log_dir)
es = EarlyStopping(monitor='val_acc', patience=20)
mc = ModelCheckpoint(log_dir + '\\CIFAR10-EP{epoch:02d}-ACC{val_acc:.4f}.h5',
						 monitor='val_acc', save_best_only=True)
tb = TensorBoard(log_dir=log_dir, histogram_freq=1,  write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

history = model.fit(X_train,Y_train,
				batch_size = BATCH_SIZE,
				epochs = NB_EPOCH,
				verbose=VERBOSE,
				validation_split=VALIDATION_SPLIT,
				callbacks=[es, mc ,tb]
				)

score = model.evaluate(X_test,Y_test,verbose=VERBOSE)
print("")
print("====================================")
print("====================================")
print(score[0])
print(score[1])
print("====================================")
print("====================================")

#save model
model.save("./CIFAR10_MODEL"+str(score[1])+".h5")

#show the data in history
print(history.history.keys())

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

rand_id = np.random.choice(range(10000), size=10)
X_pred = np.array([X_test[i] for i in rand_id])
y_true = [Y_test[i] for i in rand_id]
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