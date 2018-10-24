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
import functools
import random 

gpu_number = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
print("GPU_Number:",gpu_number)

epochs = 100
opt = 'sgd'

save_dir = os.path.join(os.getcwd(), 'saved_models')

print(os.getcwd())
cur_fold = os.getcwd().split('/')[-1]
data_path = 'bzsl/' + cur_fold + '/'

try:
    model_name = sys.argv[1]
    width = sys.argv[2]
except:
    model_name = 'resnet50'
    width = '224'

seed = 333

data_root = r'bzsl/all/'
num_classes = len(os.listdir(data_root + 'train'))

retrain = False 
if retrain:
   weight = None
else:
   weight = 'imagenet'

input_tensor = Input(shape=(int(width), int(width), 3))

if model_name == 'resnet50':
    base_model = ResNet50(input_tensor=input_tensor,weights=weight,include_top=False)
    batch_size = 64 * gpu_number
elif model_name == 'xception':
    base_model = Xception(input_tensor=input_tensor,weights=weight,include_top=False)
    if width == 224:
        batch_size = 96 * gpu_number
    elif width == 299:
        batch_size = 48 * gpu_number 
    else:
        batch_size = 32 * gpu_number

if retrain:
    save_model_name = model_name + '_' + width + '_' + str(seed) + '_retrain.h5'
else:
    save_model_name = model_name + '_' + width + '_' + str(seed) + '.h5'

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu',name='dense1024')(x)
#x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax',name='last-layer')(x)

'''
with tf.device("/cpu:0"):
   model = Model(inputs=base_model.input, outputs=predictions)
'''

if gpu_number > 1:
    with tf.device("/cpu:0"):
       model = Model(inputs=base_model.input, outputs=predictions)
    model = multi_gpu_model(model, gpus=gpu_number)
else:
    model = Model(inputs=base_model.input, outputs=predictions)

for i, layer in enumerate(model.layers):
    print(i, layer.name)

'''
for layer in base_model.layers:
    layer.trainable = False
'''
'''
for layer in model.layers[:100]:
   layer.trainable = False
for layer in model.layers[100:]:
   layer.trainable = True
'''

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)
sgd = optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy,k = 3)
top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy,k = 5)
top3_acc.__name__ = 'top3_acc'
top5_acc.__name__ = 'top5_acc'

if opt == 'adam':
#if opt == 'adam' or retrain:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',top3_acc,top5_acc])
else:
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',top3_acc,top5_acc])

print('Using real-time data augmentation.')
train_datagen = ImageDataGenerator(
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
        rescale=1./255,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

valid_gen = ImageDataGenerator(rescale=1./255)

def count_number(path):
    cnt = 0
    for cur in os.listdir(path):
        if os.path.isdir(path + cur):
           cnt += len(os.listdir(path + cur))
    return cnt

train_number = count_number(data_root + 'train/')
valid_number = count_number(data_root + 'valid/')
print (train_number,valid_number)

classes=list(range(num_classes))
for i,c in zip(range(num_classes),classes):
    classes[i] = str(c)

train_generator = train_datagen.flow_from_directory(
    data_root + 'train',
    target_size=(int(width),int(width)),
    batch_size=batch_size,
    interpolation='bilinear',
    classes=classes
)
valid_generator = valid_gen.flow_from_directory(
   data_root + 'valid',
   target_size=(int(width),int(width)),
   batch_size=batch_size,
   seed=1000,
   interpolation='bilinear',
   classes=classes
)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, save_model_name)

checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
early_stopping = EarlyStopping(monitor='val_acc', patience=15, verbose=1,mode='max')

if opt == 'adam':
    callbacks_list = [checkpoint,early_stopping]
else:
    callbacks_list = [checkpoint,early_stopping,lrate]

model.fit_generator(train_generator,epochs=epochs,validation_data=valid_generator,validation_steps=valid_number // batch_size,
                        steps_per_epoch=train_number // batch_size, 
                        callbacks=callbacks_list,
                        workers=4 * gpu_number)
