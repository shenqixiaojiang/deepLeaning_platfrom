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
sys.path.append('./')
import h5py
from keras.utils import Sequence
from keras.applications.densenet import DenseNet121,DenseNet169,DenseNet201
#from skimage.transform import resize
#from skimage import exposure
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance, ImageFilter
import tensorflow as tf
import random
import functools
import random 
import gc,argparse
#from se_model.SEResNeXt import SEResNeXt
#from se_model.se_resnet import SEResNet50,SEResNet101,SEResNet154
#from se_model.se_inception_v3 import SEInceptionV3

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='valid', help='[valid,predict]')
parser.add_argument('--model', default='xception',help='[xception,resnet50,...]')
parser.add_argument('--width', default=32,type=int,help='width')
parser.add_argument('--start', default=0,type=int,help='start_channel')
parser.add_argument('--cn', default=3,type=int,help='channel_number')
parser.add_argument('--data_path', default='./data/',help='start_channel')

opt = parser.parse_args()
print(opt)

gpu_number = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
print("GPU_Number:",gpu_number)

epochs = 100
optimizer = 'sgd'

data_save_path = r'./'
save_dir = os.path.join(data_save_path, 'saved_models')

print(os.getcwd())
cur_fold = os.getcwd().split('/')[-1]

data_path = data_save_path 

model_name = str(opt.model)
width = str(opt.width)
channel_number = opt.cn
start = opt.start

'''
fid = h5py.File('data/training.h5','r')
s1 = np.array(fid['sen1'])
s2 = np.array(fid['sen2'])
label = np.array(fid['label'])
s = np.concatenate((s1,s2),axis=3)[:,:,:,start : start + channel_number]
num_classes = label.shape[1]
'''

fid_test = h5py.File('data/validation.h5','r')
s1_test = np.array(fid_test['sen1'])
s2_test = np.array(fid_test['sen2'])
s_test = np.concatenate((s1_test,s2_test),axis=3)[:,:,:,start : start + channel_number]
label_test = np.array(fid_test['label'])
num_classes = label_test.shape[1]

seed = 333
#x_train, x_test, y_train, y_test = train_test_split(s,label,test_size=0.1,random_state=seed)
x_train, x_test, y_train, y_test = train_test_split(s_test,label_test,test_size=0.2,random_state=seed,stratify=label_test)

'''
x_train = s
y_train = label
x_test = s_test
y_test = label_test
'''

if int(width) > 32:
    x_train = np.resize(x_train,(x_train.shape[0],int(width),int(width),x_train.shape[-1]))
    x_test = np.resize(x_test,(x_test.shape[0],int(width),int(width),x_test.shape[-1]))

print x_train.shape,y_train.shape,x_test.shape,y_test.shape
assert x_train.shape[0] == y_train.shape[0]
assert x_test.shape[0] == y_test.shape[0]
print 'num_classes: ', num_classes

if channel_number != 3:
    retrain = None
else:
    retrain = 'imagenet'

'''
train_data = np.load(data_path + 'train_' + width + '.npy')
train_label = np.load(data_path + 'train_label.npy')
print 'origin_data : ', train_data.shape,train_label.shape

num_classes = len(list(set(train_label)))
print('cur num_classes',num_classes)
print(train_data.shape,train_label.shape)

seed = 333
x_train, x_test, y_train, y_test = train_test_split(train_data,train_label,test_size=0.2,random_state=seed)

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
'''

input_shape = (int(width), int(width), x_train.shape[-1])
input_tensor = Input(shape=input_shape)

if model_name == 'resnet50':
    base_model = ResNet50(input_tensor=input_tensor,weights=retrain,include_top=False)
    batch_size = 48 * gpu_number
elif model_name == 'inception_v3':
    base_model = InceptionV3(input_tensor=input_tensor,weights=retrain,include_top=False)
    batch_size = 48 * gpu_number
elif model_name == 'xception':
    base_model = Xception(input_tensor=input_tensor,weights=retrain,include_top=False)
    if int(width) > 200:
        batch_size = 48 * gpu_number
    else:
        batch_size = 24 * gpu_number
    if int(width) > 299 and int(width) < 500:
        batch_size = 48 * gpu_number
    elif int(width) >= 500:
        batch_size = 32 * gpu_number
elif model_name == 'inception_resnet':
    base_model = InceptionResNetV2(input_tensor=input_tensor,weights=None,include_top=False)
    batch_size = 32 * gpu_number
elif model_name == 'nasnet':
    base_model = NASNetLarge(input_tensor=input_tensor,weights=None,include_top=False)
    batch_size = 24 * gpu_number
elif model_name == 'densenet121':
    base_model = DenseNet121(input_tensor=input_tensor,weights=None,include_top=False)
    batch_size = 48 * gpu_number
elif model_name == 'densenet169':
    base_model = DenseNet169(input_tensor=input_tensor,weights=None,include_top=False)
    batch_size = 48 * gpu_number
elif model_name == 'densenet201':
    base_model = DenseNet201(input_tensor=input_tensor,weights=None,include_top=False)
    batch_size = 32 * gpu_number
elif model_name == 'seresnext':
    base_model = SEResNeXt(input_tensor=input_tensor,num_classes=num_classes,include_top=False).model
    batch_size = 4 * gpu_number
elif model_name == 'seresnet50':
    base_model = SEResNet50(input_shape=input_shape,include_top=False)
    batch_size = 32 * gpu_number
elif model_name == 'seresnet101':
    base_model = SEResNet101(input_shape=input_shape,include_top=False)
    batch_size = 32 * gpu_number
elif model_name == 'seresnet154':
    base_model = SEResNet154(input_shape=input_shape,include_top=False)
    batch_size = 32 * gpu_number
elif model_name == 'seinception_v3':
    base_model = SEInceptionV3(input_shape=input_shape,include_top=False)
    batch_size = 32 * gpu_number

if retrain == None:
    save_model_name = model_name + '_' + width + '_' + str(start) + '_' + str(channel_number) + '_retrain.h5'
else:
    save_model_name = model_name + '_' + width + '_' + str(start) + '_' + str(channel_number) + '.h5'

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu',name='dense1024')(x)
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

#for i, layer in enumerate(model.layers):
 #   print(i, layer.name)

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
    initial_lrate = 0.01
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

if optimizer== 'adam':
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',top3_acc,top5_acc])
else:
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',top3_acc,top5_acc])

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
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
datagen.fit(x_train)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, save_model_name)

checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
early_stopping = EarlyStopping(monitor='val_acc', patience=15, verbose=1,mode='max')

if optimizer == 'adam':
    callbacks_list = [checkpoint,early_stopping]
else:
    callbacks_list = [checkpoint,early_stopping,lrate]

model.fit_generator(datagen.flow(x_train,y_train,batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks_list,
                        workers=4 * gpu_number)
