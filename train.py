import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.regularizers import l2


lines = []
# log_file = "./data/driving_log.csv"
# log0 = "./driving_log0.csv"

log_files = []
# for i in range(0, 3):
#     lf = "./driving_log{}.csv".format(i)
#     print("log {} {}".format(i, lf))
#     log_files.append(lf)

import glob, os
os.chdir(".")
for file in glob.glob("*.csv"):
    print(file)
    log_files.append(file)

for log_file in log_files:
    zero_steering_count = 0
    with open(file=log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if float(line[3]) == 0:
                if zero_steering_count == 30:
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
    # filename = source_path.split('/')[-1]
    # current_path = './data/IMG/' + filename
    image = cv2.imread(source_path)
    images.append(image)
    measurement = line[3]
    measurements.append(float(measurement))
    # flip
    images.append(cv2.flip(image, 1))
    measurements.append(float(measurement)*-1.0)

X_train = np.array(images)
y_train = np.array(measurements)

# model
# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# model.add(Flatten(input_shape=(160, 320, 3)))
# model.add(Dense(1))

# input_shape = (64,64,3)
input_shape = (160, 320, 3)
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(80, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(40, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(16, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer = l2(0.001)))
model.add(Dense(1, W_regularizer = l2(0.001)))
adam = Adam(lr = 0.0001)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
model.summary()

# model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)

print('Save Modle')
model.save('model.h5')
