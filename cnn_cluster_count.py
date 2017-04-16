import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import keras
import random
## Set seed for reproducibility
random.seed(123)
np.random.seed(123)
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator

datapath = "/home/jincheng/projects/flow_cytometry/sim_data/"

annotation = pd.read_json(os.path.join(datapath, "annotation_o10_equal.js"))

f = h5py.File(os.path.join(datapath, "sim_data_o10_equal.hdf5"), 'r')

x = []
i = 0
next = 0
for name in annotation['name']:
    if i >= next:
        print("Loading " + str(i) + " samples")
        next += 1000
    dset = np.array(f[name])
    image = dset.reshape(np.append(1, dset.shape))
    i+=1
    x.append(image)
    
x = np.concatenate(x, axis=0)      
print(x.shape)


x = x.astype('float32')
y = np.array(annotation['y'])
## Minus 1 and convert to categorical 
## 0 - 1 cluster, 1 - 2 clusters, so on
y -= 1

train_idx = random.sample(range(len(annotation)), int(0.6 * len(annotation)))
test_idx = [index for index in range(len(annotation)) if index not in train_idx]
x_train = np.take(x, train_idx, axis = 0)
x_test = np.take(x, test_idx, axis = 0)
y_train = y[train_idx]
y_test = y[test_idx]

## Conver to 
print(np.unique(y))
num_classes = len(np.unique(y))
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

print(x_train.shape)
print(y_train.shape)


# In[30]:

batch_size = 60
epochs = 40

model = Sequential()

model.add(Conv2D(16, (6, 6), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(16, (6, 6)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (6, 6)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
opt = optimizers.rmsprop(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


print(model.summary())

# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=True,  # apply ZCA whitening
#     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#     width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=False,  # randomly flip images
#     vertical_flip=False)  # randomly flip images

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
# datagen.fit(x_train)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test), verbose=1,
          shuffle=True)


# In[ ]:

y_pred = model.predict_classes(x_train)
y_true = y[train_idx]

print("\n")
print(confusion_matrix(y_true, y_pred))
print("\n")

y_pred = model.predict_classes(x_test)
y_true = y[test_idx]

print("\n")
print(confusion_matrix(y_true, y_pred))

# Save the keras model
model.save(os.path.join(datapath, 'model_3.h5'))

