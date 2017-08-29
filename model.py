import csv
import numpy as np
import cv2
from PIL import Image, ImageDraw
import sklearn
from random import shuffle

data_path =['../data/','../track1/','../bridge/']  #add new data path to this var
data_num_upper_index = -1  # -1 load all data;

# generator function used for fixing out of memory issue.
def generator(samples, batch_size=  64):
    # print('Use generator....')
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            measurements = []
            for batch_sample in batch_samples:
                # load center/left/right images
                for i in range(3):
                    current_path = batch_sample[i]
                    # print(['generator imread current_path:',current_path])
                    image = cv2.imread(current_path)
                    images.append(image)
                    # print(['generator imread:',image.shape])

                # load car operation data
                measurement =[]
                correction = 0.2
                steering = float(batch_sample[3])
                steering_left = steering + correction
                steering_right = steering - correction 
                throttle =  float(batch_sample[4])
                brake =  float(batch_sample[5])
                speed =  float(batch_sample[6])
                measurement.extend([steering,steering_left,steering_right])
                measurements.extend(measurement)
            augmented_images,augmented_measurements = flipping_images_meas(images,measurements)
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            # print()
            # print('augmented_images')
            # print(len(augmented_images))
            # print('Use generator....')
            # print('augmented_measurements')
            # print(len(augmented_measurements))

            yield sklearn.utils.shuffle(X_train, y_train)

# augment data set by flipping images and measurements
def flipping_images_meas(images,measurements):
    augmented_images = []
    augmented_measurements = []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)
    return augmented_images,augmented_measurements


print()
print('------------------------------------------load data-----------------------------------------------------------')
lines = []
images = []
measurements = []

# extract samples from log file
lines = []
for path in data_path:
    print()
    log_path = path+'driving_log.csv'
    print('loading data from '+log_path)

    with open(log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            for i in range(3):
                source_path = line[i]
                filename = source_path.split('/')[-1]  #extract file name
                line[i] = path +'IMG/' + filename  
            lines.append(line)
        print(path)
    print(len(lines))

print()
print(lines[0:1])
print(len(lines))



from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:data_num_upper_index], test_size=0.2)

# compile and train the model using the generator function
batch_size = 128
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)




print('------------------------------------------Set up NN graphic-----------------------------------------------------------')
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
# model.add(Dropout(0.5))
model.add(Dense(100))
# model.add(Dropout(0.5))
# model.add(Dense(50,activation = 'relu'))
model.add(Dense(50))
# model.add(Dropout(0.5))
model.add(Dense(1))
model.summary()
print('------------------------------------------Start training-----------------------------------------------------------')
model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples)/batch_size, validation_data=validation_generator,
            nb_val_samples=len(validation_samples)/batch_size, nb_epoch=3,verbose=1)


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
# plt.show()
# """
