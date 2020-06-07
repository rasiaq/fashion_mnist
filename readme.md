# Fashion-MNIST

## Introduction

`Fashion-MNIST` is a dataset [Zalando](https://jobs.zalando.com/tech/)'s article images.

The Fashion-MNIST dataset includes the following data:

* training set of 60,000 examples
* test set of 10,000 examples

Each example is 28x28 single channeled, grayscale image, associated with one of then following classes:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |


The goal is to develop convolutional neural network for clothing classification.

## Methods
### Data preprocessing

At first, all the images and labels have been preprocessed by following steps:

1. Each training image has been set to numpy array with `dtype='float32'`
2. The training labels have been converted to binary class vector

Secondly, from all the training data, I've separated four sets:

1. x_train - training set of images, containing 80% of all images
2. x_val - validation set of images, containing 20% of all images
3. y_train - training set of labels, containing 80% of all labels
4. y_val - validation set of labels, containing 20% of all labels

### Model

I decided to develop Sequential model, which represents linear stack of layers

#### First version of model

```python
def get_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(img_width, img_height, 1)))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
```

There are two convolutional layers, first with 64 filters with 3x3 size, and second
with 64 filters also with 3x3 size. In both layers, the activation function is `relu`.
After each convolutional layer, there is a pooling layer with kernel size 2x2.
Then, outputs from previous layers are being flattened into a vectors and fed into
two fully connected layers with dropout between them.

Since I'm dealing with multi-classes output
my loss function is going to be cross entropy function and my accuracy metric
is going to be regular accuracy which calculates how often prediction was equal
to actual output label. I chose `adam` for optimization algorithm.

#### Improved version of model

```python
def get_model():
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
```

Unfortunately, the performance of the model wasn't the best - after 5-7th epoch
training accuracy stoped growing and validation loss was increasing drastically
(The plots are placed in results section).
To avoid overfitting I've decided to put some dropout layers - one after each
convolutional layer.

#### Final model architecture

![model_architecture](https://ibb.co/yPcCyNn)

### Training

```python
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

        if score[0] < lowest_loss:
            lowest_loss = score[0]
            best_acc = score[1]
            best_model = model

    best_model.save('best_model.h5')
```

Training of one model takes 50 epochs. I've decided to run four iterations, each
time splitting training data differently, to get four models and pick the one
that gives the lowest loss value. After finished training, the best model is
saved to a file with format `.h5`
