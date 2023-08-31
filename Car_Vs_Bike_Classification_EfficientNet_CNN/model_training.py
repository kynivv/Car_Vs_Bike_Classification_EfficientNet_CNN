# Libraries
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from glob import glob
from zipfile import ZipFile
import cv2
from keras.callbacks import ModelCheckpoint

# Constants and Hyperparameters
EPOCHS = 10
BATCH_SIZE = 10

IMG_SIZE = 300
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
SPLIT = 0.25


## Dataset Extraction
#with ZipFile('cars and bikes.zip') as zippfille:
#    zippfille.extractall()


# Data Preprocessing
X = []
Y = []

data_path = 'Car-Bike-Dataset'

classes = os.listdir(data_path)

for i, name in enumerate(classes):
    images = glob(f'{data_path}/{name}/*')
    
    for image in images:
        img = cv2.imread(image)
        
        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)

X = np.asarray(X)
Y = pd.get_dummies(Y)


# Data Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size= SPLIT,
                                                    random_state= 24,
                                                    shuffle= True
                                                    )


# Model Checkpoint
checkpoint = ModelCheckpoint('output/model.h5',
                             monitor= 'val_accuracy',
                             verbose= 1,
                             save_best_only= True,
                             save_weights_only= False
                             )


# Creating Model Based on EfficientNet
base_model = keras.applications.EfficientNetB3(input_shape= IMG_SHAPE,
                                               include_top= False,
                                               pooling= 'max'
                                               )

model = keras.Sequential([
    base_model,
    layers.Dropout(0.1),

    layers.Dropout(0.15),
    layers.Dense(128, activation= 'relu'),

    layers.Dropout(0.1),
    layers.Dense(2, activation= 'softmax')
])

model.compile(optimizer= 'adam',
              metrics= ['accuracy'],
              loss= 'binary_crossentropy'
              )


# Model Training
model.fit(X_train, Y_train,
          batch_size= BATCH_SIZE,
          epochs= EPOCHS,
          callbacks = checkpoint,
          validation_data = (X_test, Y_test),
          shuffle = True,
          verbose= 1
          )