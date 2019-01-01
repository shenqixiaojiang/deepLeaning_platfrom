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
from models.xception import xception
from models.inception_v4 import inceptionv4
from models.cifar_seresnet import se_resnet18,se_resnet34,se_resnet20,se_resnet32
from models.inception_resnet_v2 import inceptionresnetv2
from models.se_inception import se_inception_v3
from models.senet import se_resnet50,se_resnet101,se_resnet152,se_resnext50_32x4d,se_resnext101_32x4d,senet154
from models.nasnet import nasnetalarge
from models.pnasnet import pnasnet5large
from models.polynet import polynet
from models.dpn import dpn98,dpn107,dpn92
from  torch.nn import CrossEntropyLoss
import logging
from PIL import Image
import numpy as np
import h5py
from glob import glob
import random,argparse
from collections import OrderedDict
import os

def load_model(model_name='resnet50',resume='Best',start_epoch=0,cn=3,
               save_dir='saved_models/',width=32,start=8,cls_number=10,avg_number=1,gpus=[0,1,2,3,4,5,6,7],kfold = 1,model_times=0,train=True):
    
    load_dict = None
    #load_dict = True if cn == 3 else None

    if model_name == 'resnet50':
        model = resnet50(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'resnet101':
        model = resnet101(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'resnet152':
        model = resnet152(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'densenet161':
        model = densenet161(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'xception':
        model = xception(num_classes=cls_number,pretrained=load_dict)
        model.conv1 = nn.Conv2d(cn, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    elif model_name == 'inception_v3':
        model = inception_v3(num_classes=cls_number,pretrained=load_dict)
        model.Conv2d_1a_3x3.conv = nn.Conv2d(cn, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    elif model_name == 'seinception_v3':
        model = se_inception_v3(num_classes=cls_number)
        model.model.Conv2d_1a_3x3.conv = nn.Conv2d(cn, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    elif model_name == 'inception_v4':
        model = inceptionv4(num_classes=cls_number,pretrained=load_dict)
        model.features[0].conv = nn.Conv2d(cn, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    elif model_name == 'inceptionresnetv2':
        model = inceptionresnetv2(num_classes=cls_number,pretrained=load_dict)
        model.conv2d_1a.conv = nn.Conv2d(cn, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    elif model_name == 'seresnet50':
        model = se_resnet50(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'seresnet101':
        model = se_resnet101(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'seresnet152':
        model = se_resnet152(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'seresnext50':
        model = se_resnext50_32x4d(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'seresnext101':
        model = se_resnext101_32x4d(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'resnet50-101':
        model = SimpleNet()
    elif model_name == 'seresnet20':
        model = se_resnet20(num_classes=cls_number)
    elif model_name == 'seresnet32':
        model = se_resnet32(num_classes=cls_number)
    elif model_name == 'seresnet18':
        model = se_resnet18(num_classes=cls_number)
    elif model_name == 'seresnet34':
        model = se_resnet34(num_classes=cls_number)
    elif model_name == 'senet154':
        model = senet154(num_classes=cls_number,pretrained=load_dict)
        model.layer0.conv1 = nn.Conv2d(cn, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    elif model_name == 'nasnet':
        model = nasnetalarge(num_classes=cls_number,pretrained=load_dict)
        model.conv0.conv = nn.Conv2d(cn, 96, kernel_size=(3, 3), stride=(2, 2), bias=False)
    elif model_name == 'dpn98':
        model = dpn98(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'dpn107':
        model = dpn107(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'dpn92':
        model = dpn92(num_classes=cls_number,pretrained=load_dict)
    elif model_name == 'polynet':
        model = polynet(num_classes=cls_number,pretrained=load_dict)
        model.stem.conv1[0].conv = nn.Conv2d(cn, 32, kernel_size=(3, 3), stride=(2, 2), bias=False) 
    elif model_name == 'pnasnet':
        model = pnasnet5large(num_classes=cls_number,pretrained=load_dict)
        model.conv_0.conv = nn.Conv2d(cn, 96, kernel_size=(3, 3), stride=(2, 2), bias=False) 
    
    #print(model)

    if '-' not in model_name and load_dict != True:
      if model_name in ['dpn98',]:
        model.features.conv1_1.conv = nn.Conv2d(cn, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      elif model_name in ['dpn92',]:
        model.features.conv1_1.conv = nn.Conv2d(cn, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      elif model_name in ['seresnet20','seresnet32']:
        model.conv1 = nn.Conv2d(cn, 16, kernel_size=3, stride=1, padding=1, bias=False)
      elif model_name in ['seresnet18','seresnet34']:
        model.conv1 = nn.Conv2d(cn, 64, kernel_size=7, stride=2, padding=3, bias=False)
      elif 'seresnext' in model_name:
        model.layer0.conv1 = nn.Conv2d(cn, 64, kernel_size=7, stride=2, padding=3, bias=False)
      elif 'seresnet' in model_name:
        model.layer0.conv1 = nn.Conv2d(cn, 64, kernel_size=7, stride=2, padding=3, bias=False)
      elif 'resnet' in model_name:
        model.conv1 = nn.Conv2d(cn, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #model.fc = torch.nn.Linear(model.fc.in_features,cls_number)
      elif 'densenet' in model_name:
        model.features.conv0 = nn.Conv2d(cn, 96, kernel_size=7, stride=2, padding=3, bias=False)

      model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    else:
      pass

    #print(model)

    load_model = False
    #if model_name == 'resnet50':
    if load_dict != True and model_name == 'resnet50' and 0:
       base_model = resnet50(pretrained=True)
       model_dict = model.state_dict()
       new_state_dict = OrderedDict()
       for k, v in base_model.state_dict().items()[1:-2]:
           new_state_dict[k] = v
       model_dict.update(new_state_dict)
       model.load_state_dict(model_dict)
       print 'load imagenet'
       load_model = True
    
    model_ = model_name + '_' + \
                   str(width) + '_' + str(start) + '_' + str(cn)
    if kfold > 1:
       model_prefix = save_dir + str(model_times) + '_' + model_
    else:
       model_prefix = save_dir + model_
    
    if resume == 'Best' and avg_number >= 1:
        weight_path = glob(model_prefix + '*pth')
        cur_index = np.argsort(-np.array([float(cur_p.split('/')[-1].split('[')[-1].split(']')[0]) for cur_p in weight_path]))
        new_state_dict = OrderedDict()
        if len(weight_path) == 0:
            resume = ''
        elif avg_number == 1:
            resume = weight_path[0]
        else:
          for cnt,index in zip(range(avg_number),cur_index[:avg_number]):
            cur_resume = weight_path[index]
            print cur_resume
            model.load_state_dict(torch.load(cur_resume))
            for k, v in model.state_dict().items():
              if cnt == 0:
                new_state_dict[k] = v
              else:
                new_state_dict[k] = new_state_dict[k] + v
              if cnt == avg_number - 1:
                new_state_dict[k] = new_state_dict[k] / float(avg_number)
          model.load_state_dict(new_state_dict)
        if train == False:
          for index in cur_index[avg_number + 2:]:
            cur_resume = weight_path[index]
            print('remove resume %s ' %cur_resume)
            os.remove(cur_resume)
    if resume != '' and avg_number == 1:
        start_epoch = int(resume.split('-')[-3])
        #print('resuming finetune from %s'%resume)
        logging.info('resuming finetune from %s'%resume)
        model.load_state_dict(torch.load(resume))

    print('start-epoch : ',start_epoch)

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
       print 'cuda_avail: True'
       if len(gpus) > 1:
           model = torch.nn.DataParallel(model,device_ids=gpus).cuda()
       else:
           model = model.cuda()
    return model,start_epoch
