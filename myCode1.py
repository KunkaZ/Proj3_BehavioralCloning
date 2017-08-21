import csv
import cv2
import numpy as np
lines = []
with open('../data/driving.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    display(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]  #extract file name
    current_path = '../data/IMG' + filename  
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)  #convert to numpy arrays
y_train = np.array(measurements)
print(X_train.shape)

#build a simply NN/ regression NN
# from keras.models import Sequential
# from keras.layers import Flatten, Dense
# model = Sequential()
# model.add(Flatten(input_shape=(170,320,3)))
# model.add(Dense(1))

# model.compile(loss='mse',optimizer='adam')
# model.fit(X_train,y_train, validation_split=0.2,shuffle=True)

# model.save('model.h5')
