import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.datasets import mnist
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.utils.np_utils import to_categorical
from keras.models import load_model
from numpy import ndarray

# save input image dimensions
img_rows, img_cols = 28, 28


def train_model() -> None:

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_index = 35
    print(y_train[image_index])
    plt.imshow(x_train[image_index], cmap='Greys')
    # plt.show()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = np.float_(x_train)
    x_test = np.float_(x_test)

    x_train /= 255.0
    x_test /= 255.0

    num_classes = 10

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(img_rows, img_cols, 1)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    batch_size = 128
    epochs = 10

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save("test_model.h5")


def use_trained_model() -> None:
    # im = imageio.imread("https://i.imgur.com/a3Rql9C.png")
    im = imageio.imread("ColorTwo.png")
    # im = imageio.read("Snowman.png")

    gray = np.dot(im[..., :3], [0.299, 0.587, 0.114])
    plt.imshow(gray, cmap=plt.get_cmap('gray'))
    plt.show()

    # reshape the image
    gray = gray.reshape(1, img_rows, img_cols, 1)

    # normalize image
    gray /= 255

    # load the model
    model = load_model("test_model.h5")

    # predict digit
    prediction = model.predict(gray)
    print(prediction.argmax())


def nums_from_dataset(number: int, num_show: int) -> None:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    for img, num in zip(x_train, y_train):
        if num == number:
            plt.imshow(img, cmap='Greys')
            plt.show()
            num_show -= 1
        if num_show <= 0:
            break


def ascii_from_dataset(number_to_print: int, skip_nums: int = 0) -> None:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    bitmap: ndarray | None = next((img for img, dataset_number in zip(x_train, y_train) if dataset_number == number_to_print and (skip_nums := skip_nums - 1) <= 0), None)
    assert bitmap is not None, f"Failed to find number '{number_to_print}' in the dataset"

    for row in bitmap:
        for pixel in row:
            if pixel > 200:
                print('##', end='')
            elif pixel > 125:
                print('**', end='')
            elif pixel > 50:
                print('--', end='')
            else:
                print('  ', end='')
        print()  # Newline


if __name__ == '__main__':
    # train_model()
    # use_trained_model()
    # nums_from_dataset(9, 8)
    for i in range(10):
        for j in range(100):
            ascii_from_dataset(i, j)
