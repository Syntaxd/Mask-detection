import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers


datagen = ImageDataGenerator (
            rescale = 1./255, 
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            )
images_dir = './New Masks Dataset/'


train_generator  =    datagen.flow_from_directory(
                             images_dir + 'Train',
                             seed=42,
                             target_size = (200,200),
                             batch_size =32 ,               
                             class_mode = 'binary',
                            )

print('Test set')
test_generator = datagen.flow_from_directory(
                             images_dir + 'Test' ,
                             seed=42, 
                             target_size = (200,200),
                             batch_size = 32 ,               
                             class_mode = 'binary',
                            )

print('Validation set')
validation_generator = datagen.flow_from_directory(
                             images_dir + 'Validation' ,
                             seed=42, 
                             target_size = (200,200),
                             batch_size = 32 ,               
                             class_mode = 'binary',
                            )


from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(200,200,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(128,activation='relu'))
#Dense layer of 128 neurons
model.add(Dense(2,activation='softmax'))
#The Final layer with two outputs for two categories

# model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])



H = model.fit(
        train_generator ,
        epochs = 15,
        validation_data = validation_generator)


print(H.history)

from matplotlib import pyplot as plt
plots = ['val_accuracy','val_loss', 'accuracy', 'loss']
co = ['red','green','blue', 'orange']
cu = 0
for what in plots:
    y1 = []
    for x in H.history[what]:
        y1.append(x)
    plt.plot(H.epoch, y1, color=co[cu],linewidth = 1, label = what)
    cu +=1
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.show() 
