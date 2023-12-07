import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.datasets import mnist
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.utils.np_utils import to_categorical
from keras.models import load_model


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


if __name__ == '__main__':
    # train_model()
    use_trained_model()
