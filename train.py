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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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


# Crop image to remove the sky and driving deck, resize to 64x64 dimension
def crop_resize(img):
    cropped = cv2.resize(img[60:140, :], (64, 64))
    return cropped


for log_file in log_files:
    zero_steering_count = 0
    with open(file=log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if float(line[3]) == 0:
                if zero_steering_count == 5:
                    lines.append(line)
                    zero_steering_count = 0
                zero_steering_count += 1

            if float(line[3]) != 0:
                lines.append(line)
                zero_steering_count = 0

images = []
measurements = []
left_right_images = []
left_right_measurements = []
correction = 0.9  # this is a parameter to tune
for line in lines:
    source_path = line[0]
    # filename = source_path.split('/')[-1]
    # current_path = './data/IMG/' + filename
    image = cv2.imread(source_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_resize(image)
    images.append(image)
    measurement = float(line[3])
    measurements.append(float(measurement))
    # flip
    if measurement != 0:
        images.append(cv2.flip(image, 1))
        measurements.append(float(measurement)*-1.0)
        #
        left_img_path = line[1]
        left_img = cv2.imread(left_img_path)
        # left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        left_img = crop_resize(image)
        left_right_images.append(left_img)
        steering_left = measurement + correction
        left_right_measurements.append(steering_left)
        #
        right_img_path = line[2]
        right_img = cv2.imread(right_img_path)
        # right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        right_img = crop_resize(right_img)
        left_right_images.append(right_img)
        steering_right = measurement - correction
        left_right_measurements.append(steering_right)


X_train = np.array(images)
y_train = np.array(measurements)
X_train, y_train = shuffle(X_train, y_train)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

X_train2 = np.array(left_right_images)
y_train2 = np.array(left_right_measurements)

X_train = np.concatenate((X_train, X_train2))
y_train = np.concatenate((y_train, y_train2))
# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# model.add(Flatten(input_shape=(160, 320, 3)))
# model.add(Dense(1))

input_shape = (64,64,3)
# input_shape = (160, 320, 3)
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
model.compile(optimizer=adam, loss='mse')
model.summary()

# model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, validation_data=(X_valid, y_valid))

print('Save Modle')
model.save('model.h5')
