import csv
import numpy as np
import cv2
from PIL import Image, ImageDraw
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    # print(line)
print('preprocess data...')
images = []
measurements = []
for line in lines[1:-1]:
    for i in range(3):
        # print(line)
        source_path = line[i]
        filename = source_path.split('/')[-1]  #extract file name
        current_path = '../data/IMG/' + filename  
        image = cv2.imread(current_path)
        images.append(image)
        # print(current_path)

    # measurements
    measurement =[]
    correction = 0.2
    steering = float(line[3])
    steering_left = steering + correction
    steering_right = steering - correction 
    throttle =  float(line[4])
    brake =  float(line[5])
    speed =  float(line[6])
    measurement.extend([steering,steering_left,steering_right])
    measurements.extend(measurement)
print(['image shape:',image.shape])
print(['total images num:',len(images)])
# print(len(measurements))


augmented_images = []
augmented_measurements = []
# Flipping Images And Steering Measurements
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)



X_train = np.array(augmented_images)  #convert to numpy arrays
y_train = np.array(augmented_measurements)

print(['training image num:',len(augmented_measurements)])
# #build a simply NN/ regression NN
# import tensorflow
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers import Cropping2D

print('create model')

model = Sequential()

model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0 - 0.5)))

# first set of CONV CONV => MaxPooling => Dropout
model.add(Convolution2D(6, 5, 5, border_mode = "valid"))
# model.add(Convolution2D(32, 32, 3, input_shape=(160, 320, 3), border_mode = "valid"))
model.add(MaxPooling2D())
#model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Convolution2D(6, 5, 5,  border_mode = "valid",))
model.add(MaxPooling2D())
#model.add(Dropout(0.7))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))

model.add(Dense(64))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))

model.add(Dense(1))
# model.add(Activation('softmax'))

# model.add(Lambda(lambda x: (x / 255.0 - 0.5), input_shape=(160,320,3)))
# model.add(Flatten())
# model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train, validation_split=0.2,shuffle=True, nb_epoch =4)

model.save('model.h5')
# """
