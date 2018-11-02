import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetLarge
from keras.applications.densenet import DenseNet121,DenseNet169,DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.layers.core import Dropout
from keras.layers import Dense, Dropout, concatenate, maximum
from keras.applications.resnet50 import preprocess_input
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split,StratifiedKFold
import os,math
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping
from keras import optimizers
from keras.models import load_model
from keras.utils import multi_gpu_model
import sys
import pandas as pd
from keras.utils import Sequence
import gc
#from skimage.transform import resize
#from skimage import exposure
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance, ImageFilter
import tensorflow as tf
import random

gpu_number = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
print("GPU_Number:",gpu_number)

num_classes = 5
epochs = 100
opt = 'sgd'
save_model = True 
save_dir = os.path.join(os.getcwd(), 'saved_models')
data_path = r'~/cloud/data/'

####################Load the data###############################
if 'cloud' in data_path:
    num_classes = 5
elif 'nongzuowu' in data_path:
    num_classes = 61

try:
    model_name = sys.argv[1]
    width = sys.argv[2]
    batch_size = int(sys.argv[3]) * gpu_number
except:
    model_name = 'xception-inception_v3'
    width = '384'
    batch_size = 12 * gpu_number

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

if train_data.shape[-1] > 3:
    train_data = train_data[:,:,:,:3]

assert len(list(set(train_label))) == num_classes 
print(train_data.shape,train_label.shape)

#x_train, x_test, y_train, y_test = train_test_split(train_data,train_label,test_size=0.2,random_state=333)

####################Train the model and predict the test###############################
cv_pred = []

n_splits = 5
seed = 4842 

skf = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)
for index,(train_index,test_index) in enumerate(skf.split(train_data,train_label)):
    print(index)

    x_train, x_test = train_data[train_index],train_data[test_index]
    y_train, y_test = train_label[train_index],train_label[test_index]

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

    ############build the model###############
    input_tensor = Input(shape=(int(width), int(width), 3))
    feature_total = []
    for cur_model_name in model_name.split('-'):
        if cur_model_name == 'resnet50':
            base_model = ResNet50(input_tensor=input_tensor,weights='imagenet',include_top=False)
        elif cur_model_name == 'inception_v3':
            base_model = InceptionV3(input_tensor=input_tensor,weights='imagenet',include_top=False)
        elif cur_model_name == 'xception':
            base_model = Xception(input_tensor=input_tensor,weights='imagenet',include_top=False)
        elif cur_model_name == 'inception_resnet':
            base_model = InceptionResNetV2(input_tensor=input_tensor,weights='imagenet',include_top=False)
        elif cur_model_name == 'nasnet':
            base_model = NASNetLarge(input_tensor=input_tensor,weights='imagenet',include_top=False)
        elif model_name == 'densenet121':
            base_model = DenseNet121(input_tensor=input_tensor,weights='imagenet',include_top=False)
        elif model_name == 'densenet169':
            base_model = DenseNet169(input_tensor=input_tensor,weights='imagenet',include_top=False)
        elif model_name == 'densenet201':
            base_model = DenseNet201(input_tensor=input_tensor,weights='imagenet',include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu',name='dense1024-' + cur_model_name)(x)
        feature_total.append(x)

    predictions = Dense(num_classes, activation='softmax',name='last-layer')\
        (concatenate(feature_total))

    if gpu_number > 1:
      with tf.device("/cpu:0"):
        model = Model(inputs=input_tensor, outputs=predictions)
      model = multi_gpu_model(model, gpus=gpu_number)
    else:
      model = Model(inputs=input_tensor, outputs=predictions)
    
    '''
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    '''
    
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

    model_full_name = model_name + '_' + width + '_' + str(seed) + '_' + str(index)
    save_model_name = model_full_name + '.h5'
    
    def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
        return lrate

    lrate = LearningRateScheduler(step_decay)
    sgd = optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)

    if opt == 'adam':
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

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

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, save_model_name)

    if save_model:
        checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1,mode='max')

    if opt == 'adam':
        callbacks_list = [early_stopping]
    else:
        callbacks_list = [early_stopping,lrate]

    if save_model:
        callbacks_list.append(checkpoint)

    model.fit_generator(datagen.flow(x_train,y_train,batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks_list,
                            workers=4 * gpu_number)
    del model
    del x_train, x_test
    gc.collect() 
    
    ############test the model###############
    test_x = np.load(data_path + 'test_' + width + '.npy')
    print test_x.shape

    x_test = test_x.astype('float32')
    x_test /= 255

    if save_model:  ##If save the model then the best acc will be restored.
        model = load_model(model_path)

    predict_sc = model.predict(x_test)
    print predict_sc.shape

    predict = np.argmax(predict_sc,axis=1)
    print(predict.shape)

    if index == 0:
        cv_pred = np.array(predict).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(predict).reshape(-1, 1)))

######## vote for the final result..#########
res_pre = []
for line in cv_pred:
    res_pre.append(np.argmax(np.bincount(line)))
res_pre = np.array(res_pre)
print(res_pre.shape)

################Get the result to submit############################
with open('submit/test_name.txt') as ff:
    test_data_list = [line.strip() for line in ff.readlines()]
    save = pd.DataFrame()
    save['filename'] = test_data_list
    save['type'] = res_pre + 1
    save.to_csv('submit/' + model_full_name + '.csv',index=False)
