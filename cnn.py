# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 06:06:13 2018

@author: Mohak
"""

#importing the libraries
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing the CNN
#same as initializing an ANN
model = Sequential()

#Adding the Layers
#1st layer would be the convolutional layer
model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#2nd layer would be pooling layer
model.add(MaxPooling2D(pool_size=(2,2), strides = 2))

#3rd step is flattening
model.add(Flatten())

#4th step is Full Connection Step
#we'll create the Ann now
#the output_dim is the no of nodes in the hidden layer
#generally, hidden nodes should be b/w input nodes and output nodes
#so a number around 100 and a power of 2 returns good results
model.add(Dense(units = 128, activation = 'relu'))
#output_dim is now known as 'units'

#output layer
#activaiton is sigmoid because we have binary output(cat or dog)
#we would have used softmax if we had multiple catagories
model.add(Dense(units = 1, activation = 'sigmoid'))
#output_dim is now known as 'units'

#compiling the CNN
#for more than 2 outputs we would have used catagorical_crossentropy
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#before we fit the model, we will use image agumentation on the images.
#it will help us to reduce the problem of overfitting
#in image agumentation, the images are tilted, shifted, flipped and rotated to generate new images. 
#This reduces the overfitting and boosts the preformance of the CNN.
#batches of images are creatred and on each batch random transformations are applied

#using the code for preprocessing (agumentation)
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'G:\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Section 8 - Building a CNN\\Convolutional_Neural_Networks\\dataset\\training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'G:\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Section 8 - Building a CNN\\Convolutional_Neural_Networks\\dataset\\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

from PIL import Image

'''for making use of GPU 
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
import theano
import keras
'''

model.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=1,
        validation_data=test_set,
        validation_steps=2000)

#save the model using .h5 extension
model.save("G:\\CNN.h5")
model.load_weights('G:\\CNN.h5')

#Test image
#for making predictions on a single image
import numpy as np
from keras.preprocessing import image
img = image.load_img(path='G:\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Section 8 - Building a CNN\\Convolutional_Neural_Networks\\dataset\\single_prediction\\cat.jpg', target_size=(64,64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
result = model.predict(img)
if result[0][0] ==1:
    print('dog')
else:
    print('cat')

