import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

class myTensorDataset(data.Dataset):
    def __init__(self, data_tensor,target_tensor,transforms):
        
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transforms = transforms
       
    def __getitem__(self, index):
        out = self.data_tensor[index]
        if self.transforms is not None:
            out = self.transforms(out)
        return out,self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)  ##The size of data_tensor



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,dim=0)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def print_info(label,predict):
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

def process_data(train,test):
    if len(train.shape) > 2:
        train_shape = train.shape
        test_shape = test.shape
        train = train.transpose((0,2,3,1)).reshape((-1,train.shape[1]))
        test = test.transpose((0,2,3,1)).reshape((-1,test.shape[1]))
    #scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(train)
    #scaler = StandardScaler().fit(train)
    scaler = MinMaxScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    if len(train_shape) > 2:
        train = train.reshape((train_shape[0],train_shape[2],train_shape[3],train_shape[1])).transpose((0,3,1,2))
        test = test.reshape((test_shape[0],test_shape[2],test_shape[3],test_shape[1])).transpose((0,3,1,2))
    return train,test

def get_mean_and_std(dataset,cn):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(cn)
    std = torch.zeros(cn)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(cn):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
