# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.layers.core import Dropout
from keras.applications.resnet50 import preprocess_input
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
import os,math
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D
from keras.layers import Concatenate, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping
from keras import optimizers
from keras.models import load_model
from keras.utils import multi_gpu_model
import sys
from keras.utils import Sequence
#from skimage.transform import resize
#from skimage import exposure
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance, ImageFilter
import tensorflow as tf
import random

gpu_number = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
print("GPU_Number:",gpu_number)

epochs = 100
opt = 'sgd'

save_dir = os.path.join(os.getcwd(), 'saved_models')

data_path = r'./zsl/all/'

try:
    model_name = sys.argv[1]
    width = sys.argv[2]
except:
    model_name = 'vgg16-hed'
    width = '224'

x_train = np.load(data_path + 'train_' + width + '.npy')
y_train = np.load(data_path + 'train_label.npy')

try:
    x_test = np.load(data_path + 'valid_' + width + '.npy')
    y_test = np.load(data_path + 'valid_label.npy')

    train_data = np.concatenate((x_train,x_test))
    train_label = np.concatenate((y_train,y_test))
except:
    train_data = x_train
    train_label = y_train

print(train_data.shape)
if train_data.shape[-1] > 3:
    train_data = train_data[:,:,:,:3]

num_classes = len(list(set(train_label)))
print('cur num_classes: ',num_classes)
print(train_data.shape,train_label.shape)
x_train, x_test, y_train, y_test = train_test_split(train_data,train_label,test_size=0.2,random_state=333)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0

#x_train = preprocess_input(x_train)
#x_test = preprocess_input(x_test)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_tensor = Input(shape=(int(width), int(width), 3),name='input')

def load_weights_from_hdf5_group_by_name(model, filepath):
    ''' Name-based weight loading '''

    import h5py

    f = h5py.File(filepath, mode='r')

    flattened_layers = model.layers
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in flattened_layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # we batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        weight_values = [g[weight_name] for weight_name in weight_names]

        for layer in index.get(name, []):
            symbolic_weights = layer.weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '") expects ' +
                                str(len(symbolic_weights)) +
                                ' weight(s), but the saved weights' +
                                ' have ' + str(len(weight_values)) +
                                ' element(s).')
            # set values
            for i in range(len(weight_values)):
                weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
                K.batch_set_value(weight_value_tuples)


def side_branch(x,stride):
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu',name='dense1024-' + str(stride))(x)
    return x

x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
b1 = side_branch(x, 1) #480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
b2= side_branch(x, 2) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x) # 120 120 128

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
b3= side_branch(x, 4) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x) # 60 60 256

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
b4= side_branch(x, 8) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x) # 30 30 512

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x) # 30 30 512
b5= side_branch(x, 16) # 480 480 1

# fuse
fuse = Concatenate(axis=-1)([b1, b2, b3, b4, b5])

# outputs
o1    = Dense(num_classes, activation='softmax',name='o1')(b1)
o2    = Dense(num_classes, activation='softmax',name='o2')(b2)
o3    = Dense(num_classes, activation='softmax',name='o3')(b3)
o4    = Dense(num_classes, activation='softmax',name='o4')(b4)
o5    = Dense(num_classes, activation='softmax',name='o5')(b5)
ofuse = Dense(num_classes, activation='softmax',name='ofuse')(fuse)

model = Model(inputs=[input_tensor], outputs=[o1, o2, o3, o4, o5, ofuse])
filepath = '~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
load_weights_from_hdf5_group_by_name(model, filepath)

batch_size = 24

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)
sgd = optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

model.compile(optimizer=sgd,
              loss={'o1': 'categorical_crossentropy',
                    'o2': 'categorical_crossentropy',
                    'o3': 'categorical_crossentropy',
                    'o4': 'categorical_crossentropy',
                    'o5': 'categorical_crossentropy',
                    'ofuse': 'categorical_crossentropy',
                    },
              loss_weights={
                  'o1': 0.2,
                  'o2': 0.2,
                  'o3': 0.2,
                  'o4': 0.2,
                  'o5': 0.2,
                  'ofuse':0.5
              },
              metrics=['accuracy'])


save_model_name = model_name + '_' + width + '.h5'

print('Using real-time data augmentation.')
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.1,  # set range for random shear
        zoom_range=0.1,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
datagen.fit(x_train)

def mygen():
    while True:
        x,y = datagen.flow(x_train,y_train,batch_size).next()
        yield [x],[y,y,y,y,y,y]

test_datagen = ImageDataGenerator()
def mygen_test():
    while True:
        x,y = next(test_datagen.flow(x_test,y_test,batch_size))
        yield [x],[y,y,y,y,y,y]

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, save_model_name)

checkpoint = ModelCheckpoint(model_path, monitor='val_ofuse_acc', verbose=1, save_best_only=True,mode='max')
early_stopping = EarlyStopping(monitor='val_ofuse_acc', patience=10, verbose=1,mode='max')

if opt == 'adam':
    callbacks_list = [checkpoint,early_stopping]
else:
    callbacks_list = [checkpoint,early_stopping,lrate]

model.fit_generator(mygen(),
                        epochs=epochs,
                        validation_data=mygen_test(),
                        callbacks=callbacks_list,
                        steps_per_epoch = int(len(x_train) // batch_size),
                        validation_steps = int(len(x_test) // batch_size),
                        workers=4 * gpu_number)

