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
number_of_epochs = 50
number_of_iterations = 4
random_state = 101  # for data splitting


def normalize_data(images, labels):
    """
    :param images: train images
    :param labels: train labels
    :return: scaled between 0-1 and reshaped train and validation images. Train labels converted to binary class
            vector.
    """

    images = np.array(images, dtype='float32')
    images = images.reshape((images.shape[0], img_width, img_height, 1))
    images /= 255

    labels = to_categorical(labels)

    return images, labels


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


def model_training(train_images, train_labels, iterations):
    """
    :param train_images: preprocessed train images
    :param train_labels: preprocessed train labels
    :param iterations: number of iterations
    :return: tuple containing best model and training history
    """
    x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels,
                                                      test_size=0.2, random_state=random_state)

    histories = []
    model = get_model()
    training_hist = [model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=number_of_epochs, verbose=0)]
    score = model.evaluate(x_val, y_val, verbose=1)
    histories.append((score[0], score[1] * 100))

    best_model = model
    index = 0
    lowest_loss = score[0]
    best_acc = score[1] * 100

    for i in range(1, iterations):
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=random_state)
        model = get_model()
        training_hist.append(model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, verbose=0))
        score = model.evaluate(x_val, y_val, verbose=0)
        histories.append((score[0], score[1] * 100))

        print(f'Loss: {score[0]}. Accuracy: {score[1] * 100} %')

        if score[0] < lowest_loss:
            best_model = model
            index = i
            lowest_loss = score[0]
            best_acc = score[1] * 100

    best_model.save('best_model.h5')

    print('All training results')
    for i in range(len(histories)):
        val_loss = histories[i][0]
        val_acc = histories[i][1]
        print(f'Model {i}. Loss: {val_loss}. Accuracy: {val_acc}')

    print(f'Lowest loss: {lowest_loss}. Accuracy: {best_acc} %')

    return best_model, training_hist[index]


def print_plots(training_hist):
    """
    :param training_hist: training history of best model
    :return:
    """
    training_acc = training_hist.history['accuracy']
    val_acc = training_hist.history['val_accuracy']

    training_loss = training_hist.history['loss']
    val_loss = training_hist.history['val_loss']

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

    axes[0].plot(training_acc, 'b')
    axes[0].plot(val_acc, 'g')
    axes[0].legend(['Training accuracy', 'Validation accuracy'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')

    axes[1].plot(training_loss, 'r')
    axes[1].plot(val_loss, 'c')
    axes[1].legend(['Training loss', 'Validation loss'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    plt.show()


def load_best_model(x_test, y_test, path='best_model.h5'):
    """
    :param x_test: preprocessed test images
    :param y_test: preprocessed test labels
    :param path: path to the best model
    :return:
    """

    model = load_model(path)
    score = model.evaluate(x_test, y_test, verbose=1)
    print(f'Final scores: \nLoss: {score[0]}\n Accuracy: {score[1] * 100}')


def main():
    train_img, train_labels = mnist_reader.load_mnist('data/fashion', kind='train')
    test_img, test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # Normalizing data
    test_img, test_labels = normalize_data(test_img, test_labels)
    train_img, train_labels = normalize_data(train_img, train_labels)

    # Running training
    _, training_history = model_training(train_img, train_labels, number_of_iterations)

    # Printing plots
    print_plots(training_history)

    # Run final test
    load_best_model(test_img, test_labels)


if __name__ == '__main__':
    main()
