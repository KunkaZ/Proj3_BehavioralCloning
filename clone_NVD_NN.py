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
print('loading data...')
images = []
measurements = []
for line in lines[1:20]:
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
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D 
from keras.layers import Cropping2D
import matplotlib.pyplot as plt

print('----create model')

# use Nvidia Architecture
model = Sequential()

# Normalizaiton and cropping
model.add(Cropping2D(cropping=((50,25),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0 - 0.5)))

# Nvidia Architecture
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),strides=(2,2),activation="relu"))
model.add(Flatten())
# model.add(Dense(1164,activation = 'relu'))
model.add(Dense(1164))
# model.add(Dense(100,activation = 'relu'))
model.add(Dense(100))
# model.add(Dense(50,activation = 'relu'))
model.add(Dense(50))
model.add(Dense(1))
model.summary()

print('----Start training')
model.compile(loss='mse',optimizer='adam')
history_object = model.fit(X_train,y_train, validation_split=0.2,shuffle=True, nb_epoch =4, verbose=1)

model.save('model.h5')
print('----model saved')

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
# """
