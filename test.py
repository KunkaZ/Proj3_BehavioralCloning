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
for line in lines[1:20]:
    # print(line)
    source_path = line[0]
    filename = source_path.split('/')[-1]  #extract file name
    current_path = '../data/IMG/' + filename  
    print(current_path)
    image = cv2.imread(current_path)
    print(image)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

print(image)
X_train = np.array(images)  #convert to numpy arrays
y_train = np.array(measurements)
xx = X_train[1]
print(X_train.shape)

# #build a simply NN/ regression NN
# import tensorflow
from keras.models import Sequential
from keras.layers import Flatten, Dense
print('create model')
model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

print('compile mode and run')
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train, validation_split=0.2,shuffle=True,epoch = 7)

model.save('model.h5')