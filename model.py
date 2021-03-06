import csv
import cv2
import numpy as np
import os

def getLinesFromDrivingLogs(dataPath):
    """
    Returns the lines from a driving log with base directory `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    """
    lines = []
    with open(dataPath+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines        

def findImages(dataPath):
    """
    Finds all the images needed for training on the path `dataPath`.
    Returns `([centerPaths], [leftPath], [rightPath], [measurement])`
    """
    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []
    lines = getLinesFromDrivingLogs(dataPath)
    center = []
    left = []
    right = []
    measurements = []
    for line in lines:
        measurements.append(float(line[3]))
        center.append(dataPath + '/IMG/' + line[0].split('\\')[-1])
        left.append(dataPath + '/IMG/' + line[1].split('\\')[-1])
        right.append(dataPath + '/IMG/' + line[2].split('\\')[-1])
    centerTotal.extend(center)
    leftTotal.extend(left)
    rightTotal.extend(right)
    measurementTotal.extend(measurements)

    return (centerTotal, leftTotal, rightTotal, measurementTotal)

def combineImages(center, left, right, measurement, correction):
    """
    Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([imagePaths], [measurements])
    """
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (imagePaths, measurements)

import sklearn

def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                image = cv2.imread(imagePath)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

def createPreProcessingLayers():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    return model

def nVidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = createPreProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    return model

# Reading images locations.
centerPaths, leftPaths, rightPaths, measurements = findImages('data')
imagePaths, measurements = combineImages(centerPaths, leftPaths, rightPaths, measurements, 0.2)
print('Total Images: {}'.format( len(imagePaths)))

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
samples = list(zip(imagePaths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# Model creation
model = nVidiaModel()

# Compiling and training the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, 
verbose=1)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

import pickle


with open('results.pickle', 'wb') as f:
    pickle.dump(history_object.history, f)

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
