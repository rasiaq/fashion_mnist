import mnist_reader
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import tensorflow as tf

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Data values
img_width = 28
img_height = 28
batch_size = 128

# Loading the data
x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# Normalizing data
x_train = np.array(x_train, dtype='float32')
x_train = x_train.reshape((x_train.shape[0], img_width, img_height, 1))
x_train /= 255

x_test = np.array(x_test, dtype='float32')
x_test = x_test.reshape((x_test.shape[0], img_width, img_height, 1))
x_test /= 255

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.Affine(
        rotate=(-20, 20),
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}
    )
], random_order=True)

images = seq(images=x_train)
x_train = np.concatenate((x_train, images))

y_train = np.concatenate((y_train, y_train))
y_train = to_categorical(y_train)


model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {}  \n Test Accuracy: {} %'.format(score[0], score[1]*100))
