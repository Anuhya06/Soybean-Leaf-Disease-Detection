import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sys
import os
from keras.applications import VGG16, ResNet50
import keras
from numpy import load
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras import backend
from keras.layers import Dense, Input, concatenate, GlobalAveragePooling2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import tensorflow as tf

"""# Image Preprocessing"""

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
traindir = "/home/iiti/Desktop/Soybean_Project/SplitKmeans/train"
validdir = "/home/iiti/Desktop/Soybean_Project/SplitKmeans/valid"
testdir = "/home/iiti/Desktop/Soybean_Project/SplitKmeans/test"


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

batch_size = 128
training_set = train_datagen.flow_from_directory(traindir,
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

valid_set = valid_datagen.flow_from_directory(validdir,
                                            target_size=(224, 224),
                                            batch_size=batch_size,
                                            class_mode='categorical')
                                            
test_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

test_set = train_datagen.flow_from_directory(testdir,
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')

class_dict = training_set.class_indices
print(class_dict)

li = list(class_dict.keys())
print(li)

train_num = training_set.samples
valid_num = valid_set.samples


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, LayerNormalization, Add
from tensorflow.keras.initializers import glorot_uniform

# Define the model
model = Sequential()

# Convolutional Block 1
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
residual1 = Add()([model.layers[-1].output, model.layers[-2].output])  # Skip connection
model.add(MaxPooling2D((2, 2)))

# Convolutional Block 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
residual2 = Add()([model.layers[-1].output, model.layers[-2].output])  # Skip connection
model.add(MaxPooling2D((2, 2)))

# Dilated Convolution
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=(2, 2)))
residual3 = Add()([model.layers[-1].output, model.layers[-2].output])  # Skip connection
model.add(MaxPooling2D((2, 2)))

# Global Average Pooling
model.add(GlobalAveragePooling2D())

# Fully Connected Layers with Dropout
model.add(Dense(512, activation='relu', kernel_initializer=glorot_uniform()))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(7, activation='softmax'))  # Assuming 7 classes in your classification task

from tensorflow.keras.callbacks import ReduceLROnPlateau

# Define a learning rate reduction callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.5,           # Reduce learning rate by a factor of 0.5
    patience=3,           # Number of epochs with no improvement before reducing LR
    min_lr=1e-6,          # Minimum learning rate
    verbose=1             # Print messages about learning rate reduction
)

# Create an optimizer with the initial learning rate
initial_lr = 0.001  # Adjust the initial learning rate as needed
optimizer = Adam(learning_rate=initial_lr)

model.summary()
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

#fitting images to CNN
history = model.fit(training_set,
                         steps_per_epoch=train_num//batch_size,
                         validation_data=valid_set,
                         epochs=50,
                         validation_steps=valid_num//batch_size,
                         callbacks=[early_stopping, reduce_lr] 
                         )
#saving model
filepath="OriginalKmeansProposed2.h5"
model.save(filepath)

train_loss, train_accuracy = model.evaluate(training_set)
print(f"Train Loss: {train_loss:.4f}")
print(f"Train Accuracy: {train_accuracy:.4f}")


val_loss, val_accuracy = model.evaluate(valid_set)
print(f"val Loss: {val_loss:.4f}")
print(f"val Accuracy: {val_accuracy:.4f}")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_set)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

"""Visualizing the Accuracy"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy of Proposed2 for Original Images')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss of Proposed2 for Original Images')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


