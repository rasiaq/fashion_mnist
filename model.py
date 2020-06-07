import mnist_reader
import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Data values
img_width = 28
img_height = 28

# CNN params
pool_size = 2
number_of_epochs = 10


def normalize_data(train_img, test_img, train_labels):
    """
    :param train_img: train images
    :param test_img: validation images
    :param train_labels: train labels
    :return: scaled between 0-1 and reshaped train and validation images. Train labels converted to binary class
            vector.
    """

    train_img = np.array(train_img, dtype='float32')
    train_img = train_img.reshape((train_img.shape[0], img_width, img_height, 1))
    train_img /= 255

    test_img = np.array(test_img, dtype='float32')
    test_img = test_img.reshape((test_img.shape[0], img_width, img_height, 1))
    test_img /= 255

    train_labels = to_categorical(train_labels)

    return train_img, test_img, train_labels


def get_model():
    """
    :return: compiled model
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 1)))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, x_train, y_train, x_val, y_val, epochs):
    """
    :param model: compiled model
    :param x_train: preprocessed train images
    :param y_train: preprocessed train labels
    :param x_val: preprocessed validation images
    :param y_val: preprocessed validation labels
    :param epochs: number of epochs
    :return:
    """

    training_history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, verbose=0)
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Test loss: {}  \n Test Accuracy: {} %'.format(score[0], score[1]*100))
    print(training_history.history.keys())

    training_accuracy = training_history.history['accuracy']
    validation_accuracy = training_history.history['val_accuracy']

    training_loss = training_history.history['loss']
    validation_loss = training_history.history['val_loss']

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

    axes[0].plot(training_accuracy, 'b')
    axes[0].plot(validation_accuracy, 'g')
    axes[0].legend(['Training accuracy', 'Validation accuracy'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')

    axes[1].plot(training_loss, 'r')
    axes[1].plot(validation_loss, 'c')
    axes[1].legend(['Training loss', 'Validation loss'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    plt.show()


def model_training(x_train, y_train, iterations):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    model = get_model()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=number_of_epochs, verbose=1)
    score = model.evaluate(x_val, y_val, verbose=1)

    best_model = model
    lowest_loss = score[0]
    best_acc = score[1]

    for i in range(1, iterations):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        model = get_model()
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, verbose=0)
        score = model.evaluate(x_val, y_val, verbose=0)

        print(f'Test loss: {score[0]}  \n Test Accuracy: {score[1]*100} %')

        if score[0] < lowest_loss:
            lowest_loss = score[0]
            best_acc = score[1]
            best_model = model

    best_model.save('best_model.h5')
    print(f'Lowest loss: {lowest_loss}. Accuracy: {best_acc} %')


def main():
    train_img, train_labels = mnist_reader.load_mnist('data/fashion', kind='train')
    test_img, test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')

    train_img, test_img, train_labels = normalize_data(train_img, test_img, train_labels)
    print(train_labels.shape)
    x_train, x_val, y_train, y_val = train_test_split(train_img, train_labels, test_size=0.2, random_state=101)

    # print('Without dropout')
    # m1 = boosted_model()
    # train_model(m1, x_train, y_train, x_val, y_val, 50)

    print('With dropout')
    m2 = get_model()
    train_model(m2, x_train, y_train, x_val, y_val, 50)


if __name__ == '__main__':
    main()
