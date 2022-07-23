#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, Activation, Flatten, Dropout, UpSampling2D
import tp1_utils as utils
import matplotlib.pyplot as plt


def task1_create_model():
	inputs = Input(shape=(64,64,3),name='inputs')

	layer = Conv2D(32, (3, 3), padding="same")(inputs)
	layer = Activation("relu")(layer)
	layer = BatchNormalization(axis=-1)(layer)

	layer = Conv2D(32, (3, 3), padding="same")(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization(axis=-1)(layer)

	layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

	layer = Conv2D(32, (3, 3), padding="same")(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization(axis=-1)(layer)

	layer = MaxPooling2D(pool_size=(2, 2 ), strides=(2, 2))(layer)

	layer = Conv2D(64, (3, 3), padding="same")(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization(axis=-1, name="last_conv")(layer)

	layer = MaxPooling2D(pool_size=(4, 4))(layer)

	features = Flatten(name='features')(layer)
	layer = Dropout(0.6)(features)
	layer = Dense(32)(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization()(layer)
	layer = Dense(10)(layer)
	layer = Activation("softmax")(layer)

	return Model(inputs = inputs, outputs = layer)

def task1():
	model = task1_create_model()
	opt = Adam(learning_rate=1e-4)
	#opt = SGD(learning_rate=1e-3, momentum=0.9)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	model.summary()
	NUM_EPOCHS = 100
	BS=32

	H = model.fit(train_X, train_classes,validation_data=(test_X, test_classes), batch_size=BS, epochs=NUM_EPOCHS)
	model.save_weights('task1_weights.h5')

	fig, ax1 = plt.subplots()

	ax1.plot(H.history['accuracy'])
	ax1.plot(H.history['val_accuracy'])
	ax1.set_ylabel("Accuracy")
	ax1.legend(['train_acc', 'validation_acc'], loc='center left')

	ax2 = ax1.twinx()

	ax2.plot(H.history['loss'], "r")
	ax2.plot(H.history['val_loss'], "g")
	ax2.set_ylabel("Loss")
	ax2.legend(['train_loss', 'validation_loss'], loc='center right')

	plt.title('Multiclass')
	plt.xlabel('Epochs')
	plt.savefig("task1.png")

def task2():
	task1_model = task1_create_model()
	opt = Adam(learning_rate=1e-4)

	layer = Dropout(0.3)(task1_model.get_layer('features').output)
	layer = Dense(32)(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization()(layer)
	layer = Dense(10)(layer)
	layer = Activation("sigmoid")(layer)

	task2_model = Model(inputs = task1_model.get_layer("inputs").output, outputs = layer)

	task2_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])
	task2_model.summary()
	NUM_EPOCHS = 100
	BS=32

	H = task2_model.fit(train_X, train_labels,validation_data=(test_X, test_labels), batch_size=BS, epochs=NUM_EPOCHS)

	fig, ax1 = plt.subplots()

	ax1.plot(H.history['binary_accuracy'])
	ax1.plot(H.history['val_binary_accuracy'])
	ax1.set_ylabel("Accuracy")
	ax1.legend(['train_acc', 'validation_acc'], loc='center')

	ax2 = ax1.twinx()

	ax2.plot(H.history['loss'], "r")
	ax2.plot(H.history['val_loss'], "g")
	ax2.set_ylabel("Loss")
	ax2.legend(['train_loss', 'validation_loss'], loc='center right')

	plt.title('Multilabel')
	plt.xlabel('Epochs')
	plt.savefig("task2.png")

def task3():
	inputs = Input(shape=(64,64,3),name='inputs')

	layer = Conv2D(4, (3, 3), padding="same")(inputs)
	layer = Activation("relu")(layer)
	layer = BatchNormalization(axis=-1)(layer)

	#layer = MaxPooling2D(pool_size=(2, 2))(layer)

	layer = Conv2D(16, (3, 3), padding="same")(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization(axis=-1)(layer)

	layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

	layer = Conv2D(16, (3, 3), padding="same")(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization(axis=-1)(layer)

	layer = MaxPooling2D(pool_size=(2, 2 ), strides=(2, 2))(layer)

	layer = Conv2D(32, (3, 3), padding="same")(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization(axis=-1, name="last_conv")(layer)

	layer = UpSampling2D(interpolation="nearest", size=(2, 2))(layer)

	layer = Conv2D(16, (3, 3), padding="same")(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization(axis=-1)(layer)

	layer = UpSampling2D(interpolation="nearest", size=(2, 2))(layer)

	layer = Conv2D(16, (3, 3), padding="same")(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization(axis=-1)(layer)

	layer = Conv2D(4, (3, 3), padding="same")(layer)
	layer = Activation("relu")(layer)
	layer = BatchNormalization(axis=-1)(layer)

	layer = Conv2D(1, (1, 1), padding="same")(layer)
	layer = Activation("sigmoid")(layer)

	task3_model = Model(inputs = inputs, outputs = layer)

	opt = Adam(learning_rate=1e-4)
	task3_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["binary_accuracy"])
	task3_model.summary()
	NUM_EPOCHS = 100
	BS=32

	H = task3_model.fit(train_X, train_masks,validation_data=(test_X, test_masks), batch_size=BS, epochs=NUM_EPOCHS)
	predicts = task3_model.predict(test_X)
	utils.compare_masks('test_compare.png',data['test_masks'],predicts)

	fig, ax1 = plt.subplots()

	ax1.plot(H.history['binary_accuracy'])
	ax1.plot(H.history['val_binary_accuracy'])
	ax1.set_ylabel("Accuracy")
	ax1.legend(['train_acc', 'validation_acc'], loc='center')

	ax2 = ax1.twinx()

	ax2.plot(H.history['loss'], "r")
	ax2.plot(H.history['val_loss'], "g")
	ax2.set_ylabel("Loss")
	ax2.legend(['train_loss', 'validation_loss'], loc='center right')

	plt.title('Semantic Segmentation')
	plt.xlabel('Epochs')
	plt.savefig("task3.png")

data = utils.load_data()

train_X = data['train_X']
test_X = data['test_X']
train_masks = data['train_masks']
test_masks = data['test_masks']
train_classes = data['train_classes']
train_labels = data['train_labels']
test_classes = data['test_classes']
test_labels = data['test_labels']

task1()
#task2()
#task3()
