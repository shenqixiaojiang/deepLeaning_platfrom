#coding=utf-8
import os
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50,resnet101,resnet152,densenet161,densenet169,densenet201,inception_v3
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from utils.train_util import train, trainlog
from utils.other_util import get_mean_and_std,myTensorDataset,process_data
from utils.data_aug import Compose,RandomHflip,RandomVflip,Normalize,Resize,RandomResizedCrop,CenterCrop,RandomRotate,RandomErasing
from models.load_model import load_model
from  torch.nn import CrossEntropyLoss
import logging
from PIL import Image
import numpy as np
import h5py
from glob import glob
import random,argparse
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', help='[valid,predict,train]')
parser.add_argument('--trainval', default='valid', help='[train,valid]')
parser.add_argument('--model', default='resnet50',help='[xception,resnet50,...]')
parser.add_argument('--width', default=32,type=int,help='width')
parser.add_argument('--batchsize', default=24,type=int,help='batch-size')
parser.add_argument('--start', default=8,type=int,help='start_channel')
parser.add_argument('--start_epoch', default=0,type=int,help='start_epoch')
parser.add_argument('--end_epoch', default=100,type=int,help='end_epoch')
parser.add_argument('--cn', default=10,type=int,help='channel_number')
parser.add_argument('--seed', default=666,type=int,help='seed to split data')
parser.add_argument('--baselr', default=0.001,type=float,help='the init learning rate')
parser.add_argument('--wd', default=0.0001,type=float,help='the init learning rate')
parser.add_argument('--optimizer', default='SGD',help='[SGD,Adam]')
parser.add_argument('--avg_number', default=1,type=int,help='avg_number to get the new model')
parser.add_argument('--kfold',default=1,type=int,help='kfold for train')
parser.add_argument('--mixup',action='store_true',default=False,help='mixup for data')
parser.add_argument('--process',action='store_true',default=False,help='process for data')
parser.add_argument('--label_smoothing',default=0.0,type=float,help='label smoothing')
parser.add_argument('--gpus', nargs='*', type=int, default=[0],help="How many GPUs to use.")
parser.add_argument('--focal_loss',action='store_true',default=False,help='focal loss')
parser.add_argument('--data_path', default='./data/',help='data-path')
parser.add_argument('--resume', default='Best',help='fine-tune model')
parser.add_argument('--save_dir', default='./saved_models/',help='dir to save model')

opt = parser.parse_args()
print(opt)

if len(opt.gpus) == 1:
    gpu_number = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    print("GPU_Number:",gpu_number)
else:
    print("GPU_Number:",len(opt.gpus))

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
random.seed(opt.seed)
torch.backends.cudnn.benchmark = True ##accelerate the training
torch.backends.cudnn.deterministic = True

batch_size = opt.batchsize
org_size = opt.width + 32
target_size = opt.width

if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)
logfile = '%s/trainlog.log'% opt.save_dir
trainlog(logfile)

def train_main(x_train,x_test,y_train,y_test,model_times=0):
    if opt.process:
        #x_train = x_train[:1000]
        #x_test = x_test[:100]
        print np.mean(x_train),np.mean(x_test),np.min(x_train),np.min(x_test),np.max(x_train),np.max(x_test)
        x_train,x_test = process_data(x_train,x_test)
        print np.mean(x_train),np.mean(x_test),np.min(x_train),np.min(x_test),np.max(x_train),np.max(x_test)

    if opt.cn == 3:
      data_transforms = {
       'train' : transforms.Compose([
                 transforms.ToPILImage(),
                 transforms.RandomRotation(degrees=45,resample=Image.BICUBIC),
                 #transforms.RandomRotation(degrees=30,resample=Image.BICUBIC),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 #transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2, hue=0.2),
                 transforms.RandomResizedCrop(target_size,scale=(0.64,1.0)),
                 #transforms.RandomResizedCrop(target_size,scale=(0.36,1.0)),
                 transforms.ToTensor(),
                 #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
       ]),
       'val': transforms.Compose([
              transforms.ToPILImage(),
              #transforms.Resize(org_size),
              #transforms.CenterCrop(target_size),
              transforms.ToTensor(),
              #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ])
      }
    else:
      data_transforms = {
        'train' : Compose([
             RandomRotate((0,45)),
             RandomHflip(),
             RandomVflip(),
             RandomErasing(),
             #Resize((target_size,target_size)),
             RandomResizedCrop((target_size,target_size)),
         ]),
         'val': Compose([
             Resize((target_size,target_size)),
             #CenterCrop((target_size,target_size)),
         ])
      }


    #traindir = r'/media/disk1/fordata/web_server/multiGPU/cccccc/cloud/train/' ##train_dir
    #train_dataset = datasets.ImageFolder(traindir,data_transforms['train'])
    #test_dataset = datasets.ImageFolder(traindir,data_transforms['val'])

    train_x = torch.stack([torch.Tensor(i) for i in x_train])
    train_y = torch.Tensor(y_train)
    #train_y = torch.stack([torch.Tensor(i) for i in y_train])

    val_x = torch.stack([torch.Tensor(i) for i in x_test])
    val_y = torch.Tensor(y_test)
    #val_y = torch.stack([torch.Tensor(i) for i in y_test])


    #train_dataset = torch.utils.data.TensorDataset(train_x,train_y)
    #valid_dataset = torch.utils.data.TensorDataset(val_x,val_y)
    train_dataset = myTensorDataset(train_x,train_y,data_transforms['train'])
    valid_dataset = myTensorDataset(val_x,val_y,data_transforms['val'])

    data_set = {}
    data_set['train'] = train_dataset
    data_set['val'] = valid_dataset

    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=16)
    dataloader['val'] = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                   shuffle=False, num_workers=16)
    ####print the mean and std of dataset
    #print (get_mean_and_std(train_dataset,cn=opt.cn))
    #print (get_mean_and_std(valid_dataset,cn=opt.cn))

    model,start_epoch = load_model(model_name=opt.model,resume=opt.resume,start_epoch=opt.start_epoch,cn=opt.cn, \
                   save_dir=opt.save_dir,width=opt.width,start=opt.start,cls_number=cls_number,avg_number=opt.avg_number, \
                   gpus=opt.gpus,model_times=model_times,kfold=opt.kfold)

    base_lr = opt.baselr
    weight_decay = opt.wd

    load_model_flag = False
    if load_model_flag:
        conv1_params = list(map(id, model.conv1.parameters()))
        fc_params = list(map(id, model.fc.parameters()))
        base_params = filter(lambda p: id(p) not in conv1_params + fc_params,model.parameters())
        optimizer = optim.Adam([{'params': base_params},{'params': model.conv1.parameters(), 'lr': base_lr * 10}, \
                                {'params': model.fc.parameters(), 'lr': base_lr * 10}
                               ], lr=base_lr, weight_decay=weight_decay, amsgrad=True)
    else:
        #optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay, amsgrad=True)
        if opt.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay, amsgrad=True)
        elif opt.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay,momentum=0.9)

    criterion = CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.33)
    #exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=4)
    #exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [10,20,30,40], gamma=0.1)
    #exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=5,eta_min=4e-08)

    iter_per_epoch = len(data_set['train']) // opt.batchsize
    #print (iter_per_epoch)
    model_name = opt.model + '_' + str(opt.width) + '_' + str(opt.start) + '_' + str(opt.cn)
    if opt.kfold > 1:
        model_name = str(model_times) + '_' + model_name
    
    train(model,
          model_name=model_name,
          end_epoch=opt.end_epoch,
          start_epoch=start_epoch,
          optimizer=optimizer,
          criterion=criterion,
          exp_lr_scheduler=exp_lr_scheduler,
          data_set=data_set,
          data_loader=dataloader,
          save_dir=opt.save_dir,
          cls_number=cls_number,
          print_inter=iter_per_epoch // 4,
          val_inter=iter_per_epoch,
          mixup=opt.mixup,
          label_smoothing=opt.label_smoothing,
          focal_loss=opt.focal_loss
          )
    torch.cuda.empty_cache()

if __name__ == '__main__':

    if opt.trainval == 'valid':
        fid_test = h5py.File(opt.data_path + 'validation.h5','r')
        s1_test = np.array(fid_test['sen1'])
        s2_test = np.array(fid_test['sen2'])
        s_test = np.concatenate((s1_test,s2_test),axis=3)[:,:,:,opt.start:opt.start + opt.cn]
        label_test = np.array(fid_test['label'])
    else:
        s_test = np.load(opt.data_path + 'clean_data.npy')[:,:,:,opt.start:opt.start + opt.cn]
        label_test = np.load(opt.data_path + 'clean_label.npy')
    s_test = s_test.transpose((0,3,1,2))
    cls_number = label_test.shape[1]
    label_test = np.argmax(label_test,axis=1)
    #label_test = np.argmax(label_test,axis=1).reshape(len(label_test),1)
    print s_test.shape,label_test.shape

    if opt.trainval == 'valid':
        if opt.kfold > 1:
            skf = StratifiedKFold(n_splits=opt.kfold,random_state=opt.seed,shuffle=True)
            for index,(train_index,test_index) in enumerate(skf.split(s_test,label_test)):
                print(str(index) + ' model-times...')
                x_train, x_test, y_train, y_test = s_test[train_index],s_test[test_index],\
                                                   label_test[train_index],label_test[test_index]
                if x_train.shape[0] % opt.batchsize == 1:
                    x_train = x_train[:-1]
                    y_train = y_train[:-1]
                print x_train.shape,x_test.shape,y_train.shape, y_test.shape
                train_main(x_train,x_test,y_train,y_test,model_times=index)
        else:
            x_train, x_test, y_train, y_test = train_test_split(s_test,label_test,test_size=0.2,random_state=opt.seed,stratify=label_test)
            print x_train.shape,x_test.shape,y_train.shape, y_test.shape
            train_main(x_train,x_test,y_train,y_test)
    else:
        x_train = s_test
        y_train = label_test
        fid_test = h5py.File(opt.data_path + 'validation.h5','r')
        s1_test = np.array(fid_test['sen1'])
        s2_test = np.array(fid_test['sen2'])
        x_test = np.concatenate((s1_test,s2_test),axis=3)[:,:,:,opt.start:opt.start + opt.cn]
        x_test = x_test.transpose((0,3,1,2))
        y_test = np.array(fid_test['label'])
        y_test = np.argmax(y_test,axis=1)

        print x_train.shape,x_test.shape,y_train.shape, y_test.shape
        train_main(x_train,x_test,y_train,y_test)
