import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
idx = 0
log_file = "./data/driving_log.csv"
# log_file = "./driving_log1.csv"
zero_steering_count = 0
with open(file=log_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if idx == 0:
            idx += 1
            continue
        if float(line[3]) == 0:
            if zero_steering_count == 6:
                lines.append(line)
                zero_steering_count = 0
            zero_steering_count += 1

        if float(line[3]) != 0:
            lines.append(line)
            zero_steering_count = 0

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = line[3]
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

# model
model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.23, shuffle=True, nb_epoch=6)

print('Save Modle')
model.save('model.h5')
