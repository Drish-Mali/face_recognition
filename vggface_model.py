import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import models
import numpy as np
from tensorflow.keras import utils
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, Activation, Dropout, Flatten
from tensorflow.python.keras.models import load_model


def model():
    vgg_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False)
    for layer in vgg_model.layers:
        layer.trainable = False
    img_width, img_height = 150, 150

    train_data_dir = "C:\\Users\\deepa\\PycharmProjects\\opencv\\dataset\\train"
    validation_data_dir = "C:\\Users\\deepa\\PycharmProjects\\opencv\\dataset\\test"
    nb_train_samples = 300
    nb_validation_samples = 100
    epochs = 5
    batch_size = 3

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    vgg_model = keras.applications.Xception(
        weights='imagenet',
        input_shape=(150, 150, 3),
        include_top=False)
    for layer in vgg_model.layers:
        layer.trainable = False

    Conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same",
                   activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=0), name='Conv1')(
        vgg_model.layers[-1].output)
    # MaxPool Layer
    Pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same", data_format='channels_last', name='Pool3')(
        Conv1)
    flat1 = Flatten()(Pool3)
    class1 = Dense(1024, activation='relu')(flat1)
    class2 = Dense(1024, activation='relu')(class1)
    class3 = Dense(256, activation='relu')(class2)
    output = Dense(4, activation='softmax')(class3)

    model = Model(inputs=vgg_model.inputs, outputs=output)
    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy',
                  metrics=["accuracy"])
    model.fit(train_generator,
              steps_per_epoch=nb_train_samples // batch_size,
              epochs=5,
              validation_data=validation_generator,
              validation_steps=nb_validation_samples // batch_size)
    model.save_weights('t.h5')
    model.save('a.h5')


def predict(file):
    m = load_model("a.h5")
    img = load_img(file, target_size=(150, 150))
    img = img_to_array(img)
    img = np.array(img).astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    p = m.predict(img)
    print(p)
    class_names = ['deepak', 'drish', 'rashila', 'rubash']
    print((p))
    return class_names[np.argmax(p)]


def result(img):
    m = load_model("a.h5")
    img = img_to_array(img)
    img = np.array(img).astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    p = m.predict(img)
    class_names = ['deepak', 'drish', 'rashila', 'rubash']
    print((p))
    return class_names[np.argmax(p)]


