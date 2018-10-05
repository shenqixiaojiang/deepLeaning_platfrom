# -*- coding: UTF-8 -*-
# Authorized  by ShenShiWei
# Created on  2018/9/12 上午11:00
# Email: shenshiwei11@163.com
# From kwai, www.kuaishou.com
# ©2015-2018 All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.applications import *
from keras.preprocessing.image import *
import h5py
import math
import gc
from keras.models import *
from keras.layers import *
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.callbacks import *
import tensorflow as tf
from keras import optimizers
from densenet161 import DenseNet161
from resnet152 import ResNet152
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import load_model
from keras.utils import multi_gpu_model

gpu_number = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
print("GPU_Number:",gpu_number)

import warnings
warnings.filterwarnings('ignore')

def d_preprocess_input(x):
    x = x[:, :, ::-1]
    x[:, :, 0] = (x[:, :, 0] - 103.94) * 0.017
    x[:, :, 1] = (x[:, :, 1] - 116.78) * 0.017
    x[:, :, 2] = (x[:, :, 2] - 123.68) * 0.017
    return x
def r_preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x

batch_size = 24 * gpu_number
ft_epoch = 100
cls_number = 61
save_dir = os.path.join(os.getcwd(), 'saved_models')

r_image_size=(224,224)
d_image_size=(299,299)
x_image_size=(299,299)
r_freeze_layer=436
d_freeze_layer=408
x_freeze_layer=66

train_dir='train_data/'
valid_dir='valid_data/'

d_weights_path = 'weights/densenet161_weights_tf.h5'
r_weights_path = 'weights/resnet152_weights_tf.h5'

r_input_tensor = Input((224, 224, 3),name='r_input')
r_model = ResNet152(input_tensor=r_input_tensor, weights='imagenet', include_top=False)
r_ = GlobalAveragePooling2D()(r_model.output)


'''
d_model = DenseNet161(reduction=0.5,weights_path=d_weights_path)
d_model.layers[0].name='d_input'
d_input_tensor=d_model.input
d_ = d_model.output
'''
d_input_tensor = Input(shape=(299, 299, 3),name='d_input')
d_model = InceptionResNetV2(input_tensor=d_input_tensor,weights='imagenet',include_top=False)
d_ = GlobalAveragePooling2D()(d_model.output)

x_input_tensor = Input((299, 299, 3),name='x_input')
x_model = Xception(input_tensor=x_input_tensor, weights='imagenet', include_top=False)
x_ = GlobalAveragePooling2D()(x_model.output)

print(r_.shape,d_.shape,x_.shape)
cc = concatenate([r_, d_, x_],axis=-1)

top_model_input=Input((int(cc.shape[-1]),))
x = Dropout(0.5)(top_model_input)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(cls_number, activation='softmax')(x)
top_model=Model(top_model_input,prediction)
#top_model.load_weights(top_weights_path)

output_tensor = top_model(cc)
for layer in r_model.layers:
    layer.name='r_'+layer.name
for layer in d_model.layers:
    layer.name='d_'+layer.name
for layer in x_model.layers:
    layer.name='x_'+layer.name
model = Model(inputs=[r_input_tensor,d_input_tensor,x_input_tensor], outputs=output_tensor)
# model.layers[-1].name='output'
#model = multi_gpu_model(model, gpus=gpu_number)
'''
for layer in r_model.layers[:r_freeze_layer]:
    layer.trainable = False
for layer in r_model.layers[r_freeze_layer:]:
    layer.trainable = True
for layer in d_model.layers[:d_freeze_layer]:
    layer.trainable = False
for layer in d_model.layers[d_freeze_layer:]:
    layer.trainable = True
for layer in x_model.layers[:x_freeze_layer]:
    layer.trainable = False
for layer in x_model.layers[x_freeze_layer:]:
    layer.trainable = True
'''

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

r_gen = ImageDataGenerator(
    preprocessing_function=r_preprocess_input
)
d_gen = ImageDataGenerator(
    preprocessing_function=d_preprocess_input
)
x_gen = ImageDataGenerator(
    preprocessing_function=xception.preprocess_input
)

classes=list(range(cls_number))
for i,c in zip(range(cls_number),classes):
    classes[i] = str(c)

r_generator = r_gen.flow_from_directory(
    train_dir,
    target_size=r_image_size,
    batch_size=batch_size,
    # shuffle=False,
    seed=1000,
    interpolation='bilinear',
    classes=classes
)
d_generator = d_gen.flow_from_directory(
    train_dir,
    target_size=d_image_size,
    batch_size=batch_size,
    # shuffle=False,
    seed=1000,
    interpolation='bilinear',
    classes=classes
)
x_generator = x_gen.flow_from_directory(
    train_dir,
    target_size=x_image_size,
    batch_size=batch_size,
    # shuffle=False,
    seed=1000,
    interpolation='bilinear',
    classes=classes
)

def mygen():
    while True:
        r=next(r_generator)
        d=next(d_generator)
        x=next(x_generator)
        x1=r[0]
        x2=d[0]
        x3=x[0]
        y=r[1]
        yield ({'r_r_input': x1, 'd_d_input': x2, 'x_x_input': x3}, y)

r_generator_v = r_gen.flow_from_directory(
    valid_dir,
    target_size=r_image_size,
    batch_size=batch_size,
    # shuffle=False,
    seed=123,
    interpolation='bilinear',
    classes=classes
)
d_generator_v = d_gen.flow_from_directory(
    valid_dir,
    target_size=d_image_size,
    batch_size=batch_size,
    # shuffle=False,
    seed=123,
    interpolation='bilinear',
    classes=classes
)
x_generator_v = x_gen.flow_from_directory(
    valid_dir,
    target_size=x_image_size,
    batch_size=batch_size,
    # shuffle=False,
    seed=123,
    interpolation='bilinear',
    classes=classes
)

def mygen_v():
    while True:
        r=next(r_generator_v)
        d=next(d_generator_v)
        x=next(x_generator_v)
        x1=r[0]
        x2=d[0]
        x3=x[0]
        y=r[1]
        yield ({'r_r_input': x1, 'd_d_input': x2, 'x_x_input': x3}, y)

three_gen = mygen()
three_gen_v = mygen_v()

model_name = '3model_fine.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1,mode='max')
callbacks_list = [checkpoint,early_stopping]

model.fit_generator(
    three_gen,
    steps_per_epoch= math.ceil(len(x_generator.filenames)/batch_size),
    validation_data= three_gen_v,
    callbacks=callbacks_list,
    epochs=ft_epoch,
    validation_steps=math.ceil(len(x_generator_v.filenames)/batch_size)
)
#model.save(model_path)

gc.collect()
