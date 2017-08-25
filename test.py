import csv
import numpy as np
import cv2
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    # print(line)

images = []
measurements = []
for line in lines[1:1000]:
    # print(line)
    source_path = line[0]
    filename = source_path.split('/')[-1]  #extract file name
    current_path = '../data/IMG/' + filename  
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
# print(line)

augmented_images = []
augmented_measurements = []
# Flipping Images And Steering Measurements
for images, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)  #convert to numpy arrays
y_train = np.array(augmented_measurements)

# #build a simply NN/ regression NN
# import tensorflow
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D 
print('create model')

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0 - 0.5), input_shape=(160,320,3)))

# model.add(Convolution2D(nb_filter=16, nb_row=7, nb_col=7, \
    # border_mode='valid', input_shape=(1, 31, 31), activation='tang'))

# first set of CONV CONV => MaxPooling => Dropout
model.add(Convolution2D(32, 32, 3, border_mode = "valid"))
# model.add(Convolution2D(32, 32, 3, input_shape=(160, 320, 3), border_mode = "valid"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(50, 10, 5,  border_mode = "valid",))
model.add(MaxPooling2D((4,4)))
model.add(Dropout(0.7))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(1))
# model.add(Activation('softmax'))

# model.add(Lambda(lambda x: (x / 255.0 - 0.5), input_shape=(160,320,3)))
# model.add(Flatten())
# model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train, validation_split=0.2,shuffle=True, nb_epoch =7)

model.save('model.h5')
