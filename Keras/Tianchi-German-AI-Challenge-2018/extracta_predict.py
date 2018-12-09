import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
import os
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import sys,argparse
import functools
from sklearn.metrics import accuracy_score
import h5py
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='valid', help='[valid,predict]')
parser.add_argument('--model', default='xception',help='[xception,resnet50,...]')
parser.add_argument('--width', default=32,type=int,help='width')
parser.add_argument('--start', default=0,type=int,help='start_channel')
parser.add_argument('--cn', default=3,type=int,help='channel_number')
parser.add_argument('--data_path', default='./data/',help='start_channel')

opt = parser.parse_args()
print(opt)

if opt.mode == 'valid':
    fid = h5py.File(opt.data_path + 'validation.h5','r')
elif opt.mode == 'predict' or opt.mode == 'test':
    fid = h5py.File(opt.data_path + 'test1.h5','r')

if opt.mode != 'train': 
    s1 = np.array(fid['sen1'])
    s2 = np.array(fid['sen2'])
    s = np.concatenate((s1,s2),axis=3)
else:
    s = np.load(opt.data_path + 'part4.npy')

if opt.mode == 'valid':
    label = np.array(fid['label'])
    label = np.argmax(label,axis=1)
elif opt.mode == 'train':
    label = np.load(opt.data_path + 'label4.npy')

data = s[:,:,:,opt.start:opt.start + opt.cn]
print data.shape

model_full_name = opt.model + '_' + str(opt.width)
print('model name is :', model_full_name)

top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy,k = 3)
top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy,k = 5)
top3_acc.__name__ = 'top3_acc'
top5_acc.__name__ = 'top5_acc'
if opt.cn == 3:
    model = load_model('saved_models/' +  model_full_name + '_' + str(opt.start) + '_'
                   + str(opt.cn) + '.h5',custom_objects={'top3_acc': top3_acc,'top5_acc': top5_acc})
else:
    model = load_model('saved_models/' +  model_full_name + '_' + str(opt.start) + '_'
                   + str(opt.cn) + '_retrain.h5',custom_objects={'top3_acc': top3_acc,'top5_acc': top5_acc})
if opt.mode == 'extract':
    if '-' in model_full_name:
      feature_model = Model(inputs=model.input, outputs=model.get_layer('concate-feature').output)
    else:
      feature_model = Model(inputs=model.input, outputs=model.get_layer('dense1024').output)

    predict_sc = feature_model.predict(x_test)
    print predict_sc.shape

    np.save(opt.data_path + model_full_name + '_' + '_feature.npy',predict_sc)

elif opt.mode == 'test' or opt.mode == 'predict':
    predict_sc = model.predict(data)
    print predict_sc.shape
    number_class = predict_sc.shape[1]
    predict = np.argmax(predict_sc,axis=1)
    print predict.shape
    submit = np.zeros(predict_sc.shape)
    for ii in range(len(predict)):
        submit[ii][predict[ii]] = 1
    submit = submit.astype(np.int8)
    #np.save(opt.data_path + 'predict-result',predict)
    #submit = keras.utils.to_categorical(predict,number_class)
    file_su = pd.DataFrame(submit)
    str_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    str_times = str_time.split('-')
    subfile_path = 'test/result_{0}_{1}_{2}_{3}_{4}.csv'.format(''.join(str_times[0:3]),''.join(str_times[3:]),model_full_name,str(opt.start),str(opt.cn))
    file_su.to_csv(subfile_path,index=False,header=None) 
    
elif opt.mode == 'valid' or opt.mode == 'train':
    predict_sc = model.predict(data)
    print predict_sc.shape
    predict = np.argmax(predict_sc,axis=1)
    print predict.shape
    print accuracy_score(label,predict)
    print classification_report(label,predict)
    
    labels = list(set(label))
    conf_mat =  confusion_matrix(label,predict,labels=labels)
    print "confusion_matrix(left labels: y_true, up labels: y_pred):"
    print "labels\t",
    for i in range(len(labels)):
        print labels[i],"\t",
    print 
    for i in range(len(conf_mat)):
        print i,"\t",
        for j in range(len(conf_mat[i])):
            print conf_mat[i][j],'\t',
        print 
    print 
