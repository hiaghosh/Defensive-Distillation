import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight')
import pickle as pkl

batch_size = 128
num_classes = 10
epochs = 50
img_rows, img_cols = 28, 28
# temperature = 20

colors = [[0,0,0], [230/255,159/255,0], [86/255,180/255,233/255], [0,158/255,115/255],
          [213/255,94/255,0], [0,114/255,178/255]]

(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

folder = os.path.abspath(os.getcwd())

def initial_model(train=True, vis=False):
    model_name = os.path.join(folder, 'initial_model.h5')
    if train:
        model = Sequential()
        model.add((Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)))
        model.add((Conv2D(32, kernel_size=(3, 3), activation='relu')))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add((Conv2D(64, kernel_size=(3, 3), activation='relu')))
        model.add((Conv2D(64, kernel_size=(3, 3), activation='relu')))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        # model.add(Lambda(lambda x: x / temperature))
        K.in_train_phase(x=model.add(Lambda(lambda x: x / 20)), alt=model.add(Lambda(lambda x: x)), training=True)
        model.add(Dense(num_classes, activation='softmax'))
        # model.summary()

        sgd = optimizers.SGD(lr=0.1, decay=0, momentum=0.5, nesterov=False)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

        with open(model_name[:-3]+'_history.h5', 'wb') as f:
            pkl.dump(history.history, f)
        model.save(model_name)

    if vis:
        with open(model_name[:-3]+'_history.h5', 'rb') as f:
            history = pkl.load(f)
        plt.plot(history['loss'], color=[0,158/255,115/255])
        plt.title('Initial Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        # plt.legend(['train'], loc='upper right')
        plt.show()

    model = load_model(model_name)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f'Original Accuracy:\t{score[1]*100}')

def distilled_model(train=True, vis=False):
    model_name = os.path.join(folder, 'distilled_model.h5')
    if train:
        initial_model = load_model(os.path.join(folder, 'initial_model.h5'))
        y_train = initial_model.predict(x_train, batch_size=batch_size)

        model = Sequential()
        model.add((Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)))
        model.add((Conv2D(32, kernel_size=(3, 3), activation='relu')))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add((Conv2D(64, kernel_size=(3, 3), activation='relu')))
        model.add((Conv2D(64, kernel_size=(3, 3), activation='relu')))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        # model.add(Lambda(lambda x: x / temperature))
        K.in_train_phase(x=model.add(Lambda(lambda x: x / 20)), alt=model.add(Lambda(lambda x: x)), training=True)
        model.add(Dense(num_classes, activation='softmax'))
        # model.summary()

        sgd = optimizers.SGD(lr=0.1, decay=0, momentum=0.5, nesterov=False)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

        with open(model_name[:-3]+'_history.h5', 'wb') as f:
            pkl.dump(history.history, f)
        model.save(model_name)

    if vis:
        with open(model_name[:-3]+'_history.h5', 'rb') as f:
            history = pkl.load(f)
        plt.plot(history['loss'], color=[0,158/255,115/255])
        plt.title('Distilled Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        # plt.legend(['train'], loc='upper right')
        plt.show()

    model = load_model(model_name)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f'Distilled Accuracy:\t{score[1]*100}')


def Main():
    initial_model(train=False, vis=True)
    distilled_model(train=False, vis=True)


if __name__ == "__main__":
    Main()
