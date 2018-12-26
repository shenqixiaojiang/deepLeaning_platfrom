import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50,resnet101,resnet152,densenet161,densenet169,densenet201,inception_v3
import torch.optim as optim
import torch.nn as nn
from models.xception import Xception
from models.senet import se_resnet50,se_resnet101,se_resnet152,se_resnext50_32x4d,se_resnext101_32x4d
from models.inception_v4 import inceptionv4
from models.inception_resnet_v2 import inceptionresnetv2
from torch.optim import lr_scheduler
from utils.train_util import train, trainlog
from utils.other_util import get_mean_and_std,myTensorDataset,process_data
from utils.data_aug import Compose,RandomHflip,RandomVflip,Normalize,Resize
from  torch.nn import CrossEntropyLoss
from models.load_model import load_model
import logging
from PIL import Image
import numpy as np
import h5py
import random,argparse
from math import ceil
from torch.autograd import Variable
from  torch.nn.functional import softmax
import sys
from glob import glob
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', help='[valid,predict,train]')
parser.add_argument('--model', default='resnet50',help='[xception,resnet50,...]')
parser.add_argument('--width', default=32,type=int,help='width')
parser.add_argument('--batchsize', default=24,type=int,help='batch-size')
parser.add_argument('--start', default=8,type=int,help='start_channel')
parser.add_argument('--cn', default=10,type=int,help='channel_number')
parser.add_argument('--avg_number', default=1,type=int,help='avg_number to get the new model')
parser.add_argument('--seed', default=666,type=int,help='seed to split data')
parser.add_argument('--start_epoch', default=0,type=int,help='start_epoch')
parser.add_argument('--crop_five',action='store_true',default=False,help='crop five for predict')
parser.add_argument('--resume', default='Best',help='fine-tune model')
parser.add_argument('--process',action='store_true',default=False,help='process for data')
parser.add_argument('--data_path', default='./data/',help='start_channel')
parser.add_argument('--load_model', default='Best',help='load_model_path')
parser.add_argument('--save_dir', default='./saved_models/',help='start_channel')

opt = parser.parse_args()
print(opt)

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
random.seed(opt.seed)
torch.backends.cudnn.benchmark = True ##accelerate the training
torch.backends.cudnn.deterministic = True

batch_size = opt.batchsize
org_size = opt.width + 32
target_size = opt.width

if not opt.crop_five:
    if opt.cn == 3:
        test_transforms = transforms.Compose([
          transforms.ToPILImage(),
          #transforms.Resize(org_size),
          #transforms.CenterCrop(target_size),
          transforms.ToTensor(),
          #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ])
    else:
        test_transforms = Compose([
          Resize((target_size,target_size)),    
        ]) 
else:
    print ('five crop...')
    test_transforms = transforms.Compose([
          transforms.ToPILImage(),
          transforms.FiveCrop(target_size),
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])

if opt.mode == 'valid':
    fid_test = h5py.File(opt.data_path + 'validation.h5','r')
elif opt.mode == 'predict' or opt.mode == 'test':
    fid_test = h5py.File(opt.data_path + 'test1.h5','r')

if opt.mode != 'train':
    s1_test = np.array(fid_test['sen1'])
    s2_test = np.array(fid_test['sen2'])
    s_test = np.concatenate((s1_test,s2_test),axis=3)[:,:,:,opt.start:opt.start + opt.cn]
    #s_test = np.concatenate((s1_test,s2_test),axis=3)[::10,:,:,opt.start:opt.start + opt.cn]
    s_test = s_test.transpose((0,3,1,2))

if opt.mode == 'valid':
    label_test = np.array(fid_test['label'])
    cls_number = label_test.shape[1]
    label_test = np.argmax(label_test,axis=1)
elif opt.mode == 'predict' or opt.mode == 'test':
    cls_number = 17
    label_test = np.ones(len(s_test))
else:
    s_test = np.load(opt.data_path + 'part4.npy')[:,:,:,opt.start:opt.start + opt.cn]
    s_test = s_test.transpose((0,3,1,2))
    label_test = np.load(opt.data_path + 'label4.npy') 
    cls_number = 17
        
print s_test.shape,label_test.shape
if opt.process:
    print np.mean(x_train),np.mean(x_test),np.min(x_train),np.min(x_test),np.max(x_train),np.max(x_test)
    x_train,x_test = process_data(x_train,x_test)
    print np.mean(x_train),np.mean(x_test),np.min(x_train),np.min(x_test),np.max(x_train),np.max(x_test) 

test_x = torch.stack([torch.Tensor(i) for i in s_test])
test_y = torch.Tensor(label_test)

data_set = {}
data_set['test'] = myTensorDataset(test_x,test_y,test_transforms)
#data_set['test'] = torch.utils.data.TensorDataset(test_x,test_y)

data_loader = {}
data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=batch_size, num_workers=8,shuffle=False, pin_memory=True)
####print the mean and std of dataset
#print (get_mean_and_std(data_set['test'],cn=opt.cn))

model,_ = load_model(model_name=opt.model,resume=opt.resume,start_epoch=opt.start_epoch,cn=opt.cn,
               save_dir=opt.save_dir,width=opt.width,start=opt.start,cls_number=cls_number,avg_number=opt.avg_number) 
model.eval()

criterion = CrossEntropyLoss()

if not os.path.exists('./result'):
    os.makedirs('./result')

print(len(data_set['test']))

test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)
test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
test_probs = np.zeros((len(data_set['test'])), dtype=np.float32)
true_label = np.zeros((len(data_set['test'])), dtype=np.int)
idx = 0
test_loss = 0
test_corrects = 0
for batch_cnt_test, data_test in enumerate(data_loader['test']):
    # print data
    if batch_cnt_test % 100 == 2:
        print("{0}/{1}".format(batch_cnt_test, int(test_size)))
    inputs, labels = data_test
    #print (inputs.size())
    inputs = Variable(inputs.cuda())
    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
    # forward
    if opt.crop_five:
        #print inputs.size()
        bs, ncrops, c, h, w = inputs.size()
        result = model(inputs.view(-1, c, h, w))
        outputs = result.view(bs, ncrops, -1).mean(1)
    else:
        outputs = model(inputs)
    
    # statistics
    if isinstance(outputs, list):
        loss = criterion(outputs[0], labels)
        loss += criterion(outputs[1], labels)
        outputs = (outputs[0]+outputs[1])/2
    else:
        loss = criterion(outputs, labels)
    pro_max, preds = torch.max(outputs, 1)
    
    test_loss += loss.item()
    #test_loss += loss.data[0]
    batch_corrects = torch.sum((preds == labels)).item()
    #batch_corrects = torch.sum((preds == labels)).data[0]
    test_corrects += batch_corrects
    test_probs[idx:(idx + labels.size(0))] = pro_max.cpu().numpy()
    test_preds[idx:(idx + labels.size(0))] = preds.cpu().numpy()
    true_label[idx:(idx + labels.size(0))] = labels.data.cpu().numpy()
    # statistics
    idx += labels.size(0)
test_loss = test_loss / test_size
test_acc = 1.0 * test_corrects / len(data_set['test'])
print('test-loss: %.4f ||test-acc@1: %.4f'
      % (test_loss, test_acc))

### submit -- result
if opt.mode == 'test':
    model_full_name = str(opt.model) + '_' + str(opt.width)
    submit = np.zeros((len(test_preds),cls_number))
    for ii in range(len(test_preds)):
        submit[ii][int(test_preds[ii])] = 1
    submit = submit.astype(np.int8)
    file_su = pd.DataFrame(submit)
    import datetime
    str_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    str_times = str_time.split('-')
    subfile_path = 'result/result_{0}_{1}_{2}_{3}_{4}.csv'.format(''.join(str_times[0:3]),''.join(str_times[3:]),model_full_name,str(opt.start),str(opt.cn))
    file_su.to_csv(subfile_path,index=False,header=None)

torch.cuda.empty_cache()
