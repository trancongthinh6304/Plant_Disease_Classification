from keras.models import Model, Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import cv2
import os
from tqdm import tqdm
import tensorflow as tf

def build_model(model_name, img_size, num_class, lr):
    if model_name == "DenseNet121":
        model = tf.keras.applications.DenseNet121(input_shape = (img_size, img_size, 3),
                                                  include_top = False,
                                                  weights = 'imagenet')
    elif model_name == "DenseNet169":
        model = tf.keras.applications.DenseNet169(input_shape = (img_size, img_size, 3),
                                                  include_top = False,
                                                  weights = 'imagenet')
    elif model_name == "DenseNet201":
        model = tf.keras.applications.DenseNet201(input_shape = (img_size, img_size, 3),
                                                  include_top = False,
                                                  weights = 'imagenet')
    elif model_name == "EfficientNetB0": 
        model = tf.keras.applications.EfficientNetB0(input_shape = (img_size, img_size, 3),
                                                     include_top = False,
                                                     weights = 'imagenet')
    elif model_name == "EfficientNetB1":
        model = tf.keras.applications.EfficientNetB1(input_shape = (img_size, img_size, 3),
                                                     include_top = False,
                                                     weights = 'imagenet')
    elif model_name == "EfficientNetB2":
        model = tf.keras.applications.EfficientNetB2(input_shape = (img_size, img_size, 3),
                                                     include_top = False,
                                                     weights = 'imagenet')
    elif model_name == "EfficientNetB3":
        model = tf.keras.applications.EfficientNetB3(input_shape = (img_size, img_size, 3),
                                                     include_top = False,
                                                     weights = 'imagenet')
    elif model_name == "EfficientNetB4":
        model = tf.keras.applications.EfficientNetB4(input_shape = (img_size, img_size, 3),
                                                     include_top = False,
                                                     weights = 'imagenet')
    elif model_name == "EfficientNetB5":
        model = tf.keras.applications.EfficientNetB5(input_shape = (img_size, img_size, 3),
                                                     include_top = False,
                                                     weights = 'imagenet')
    elif model_name == "EfficientNetB6":
        model = tf.keras.applications.EfficientNetB6(input_shape = (img_size, img_size, 3),
                                                     include_top = False,
                                                     weights = 'imagenet')
    elif model_name == "EfficientNetB7":
        model = tf.keras.applications.EfficientNetB7(input_shape = (img_size, img_size, 3),
                                                     include_top = False,
                                                     weights = 'imagenet')
    elif model_name == "InceptionResNetV2":
        model = tf.keras.applications.InceptionResNetV2(input_shape = (img_size, img_size, 3),
                                                        include_top = False,
                                                        weights = 'imagenet')
    elif model_name == "InceptionV3":
        model = tf.keras.applications.InceptionV3(input_shape = (img_size, img_size, 3),
                                                  include_top = False,
                                                  weights = 'imagenet')                                                                                                  
    elif model_name == "MobileNet":
        model = tf.keras.applications.MobileNet(input_shape = (img_size, img_size, 3),
                                                include_top = False,
                                                weights = 'imagenet')
    elif model_name == "MobileNetV2":
        model = tf.keras.applications.MobileNetV2(input_shape = (img_size, img_size, 3),
                                                  include_top = False,
                                                  weights = 'imagenet')
    elif model_name == "MobileNetV3Large":
        model = tf.keras.applications.MobileNetV3Large(input_shape = (img_size, img_size, 3),
                                                       include_top = False,
                                                       weights = 'imagenet')
    elif model_name == "MobileNetV3Small":
        model = tf.keras.applications.MobileNetV3Small(input_shape = (img_size, img_size, 3),
                                                       include_top = False,
                                                       weights = 'imagenet')
    elif model_name == "ResNet101":
        model = tf.keras.applications.ResNet101(input_shape = (img_size, img_size, 3),
                                                include_top = False,
                                                weights = 'imagenet')
    elif model_name == "ResNet101V2":
        model = tf.keras.applications.ResNet101V2(input_shape = (img_size, img_size, 3),
                                                  include_top = False,
                                                  weights = 'imagenet')
    elif model_name == "ResNet152":
        model = tf.keras.applications.ResNet152(input_shape = (img_size, img_size, 3),
                                                include_top = False,
                                                weights = 'imagenet')
    elif model_name == "ResNet152V2":
        model = tf.keras.applications.ResNet152V2(input_shape = (img_size, img_size, 3),
                                                  include_top = False,
                                                  weights = 'imagenet')
    elif model_name == "ResNet50":
        model = tf.keras.applications.ResNet50(input_shape = (img_size, img_size, 3),
                                               include_top = False,
                                               weights = 'imagenet')
    elif model_name == "ResNet50V2":
        model = tf.keras.applications.ResNet50V2(input_shape = (img_size, img_size, 3),
                                                 include_top = False,
                                                 weights = 'imagenet')
    elif model_name == "VGG16":
        model = tf.keras.applications.VGG16(input_shape = (img_size, img_size, 3),
                                            include_top = False,
                                            weights = 'imagenet')
    elif model_name == "VGG19":
        model = tf.keras.applications.VGG19(input_shape = (img_size, img_size, 3),
                                            include_top = False,
                                            weights = 'imagenet')
    elif model_name == "Xception":
        model = tf.keras.applications.Xception(input_shape = (img_size, img_size, 3),
                                               include_top = False,
                                               weights = 'imagenet')
    else:
        raise Exception("Please check your spelling or choose one of the followings: DenseNet121, DenseNet169, DenseNet201, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7, InceptionResNetV2, InceptionV3, MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small, ResNet101, ResNet101V2, ResNet152, ResNet152V2, ResNet50, ResNet50V2, VGG16, VGG19, Xception.")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

    input = Input(shape=(img_size, img_size, 3))
    x = Conv2D(3, (3, 3), padding = 'same')(input)
    x = model(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    #optimizer = Adam(lr = lr, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    optimizer = Adam(lr = lr)

    if num_class == 2:
        output = Dense(num_class, activation = 'sigmoid', name = 'root')(x)
        # model
        model = Model(input, output)
        model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    elif num_class > 2:
        output = Dense(num_class, activation = 'softmax', name = 'root')(x)
        # model
        model = Model(input, output)
        model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    else:
        raise Exception("Inappropriate num_class")

    model.summary()
    return model
