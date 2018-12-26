#coding=utf8
from __future__ import division
import torch
import os,time,datetime
import torch.nn as nn
from torch.autograd import Variable
import logging
import numpy as np
from math import ceil
import random
from utils.other_util import FocalLoss 

def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    #print('lam: ', lam)
       
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(model,
          model_name,
          end_epoch,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_set,
          data_loader,
          save_dir,
          cls_number,
          print_inter=200,
          val_inter=3500,
          mixup=False,
          label_smoothing=0.0,
          focal_loss=False
          ):

    if focal_loss:
        criterion = FocalLoss()
    
    step = -1
    val_best_acc = 0
    mid_epoch = int(end_epoch * 0.8)
    for epoch in range(start_epoch,end_epoch):
        # train phase
        exp_lr_scheduler.step(epoch)
        model.train(True)  # Set model to training mode
        for batch_cnt, data in enumerate(data_loader['train']):

            step += 1
            model.train(True)
            # print data
            inputs, labels = data

            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
            
            if label_smoothing > 0.0:
                criterion = nn.BCEWithLogitsLoss()
                #criterion = nn.KLDivLoss(size_average=False)
                #print labels.size()
                org = torch.zeros(inputs.size(0),cls_number)
                org.fill_(label_smoothing)
                for cur_ll in range(labels.size(0)):
                    org[cur_ll,labels[cur_ll]] = 1 - label_smoothing
                labels = org.cuda()
                #print labels.size() 
            mixup_flag = False
            if mixup and epoch < mid_epoch and random.random() < 0.4:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                mixup_flag = True
            
            outputs = model(inputs)
              
            if mixup_flag:
              loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
              if isinstance(outputs, list) or isinstance(outputs, tuple):  ##for inception_v3
                loss = criterion(outputs[0], labels)
                loss += criterion(outputs[1], labels)
                outputs = outputs[0]
              else:
                #print (outputs.size())
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # batch loss
            if step % print_inter == 0:
                _, preds = torch.max(outputs, 1)
                
                if label_smoothing > 0.0:
                    _ , true_label = torch.max(labels,1)
                    batch_corrects = torch.sum((preds == true_label)).item()
                else: 
                    batch_corrects = torch.sum((preds == labels)).item()
                batch_acc = batch_corrects / (labels.size(0))

                logging.info('%s [%d-%d] | batch-loss: %.3f | acc@1: %.3f'
                             % (dt(), epoch, batch_cnt, loss.data.item(), batch_acc))


            if step % val_inter == 0:
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                # val phase
                model.train(False)  # Set model to evaluate mode

                val_loss = 0
                val_corrects = 0
                val_size = ceil(len(data_set['val']) / data_loader['val'].batch_size)

                t0 = time.time()

                for batch_cnt_val, data_val in enumerate(data_loader['val']):
                    # print data
                    inputs,  labels = data_val

                    inputs = Variable(inputs.cuda())
                    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
                    if label_smoothing > 0.0:
                       criterion = nn.BCEWithLogitsLoss()
                       #criterion = nn.KLDivLoss(size_average=False)
                       #print labels.size()
                       org = torch.zeros(inputs.size(0),cls_number)
                       org.fill_(label_smoothing)
                       for cur_ll in range(labels.size(0)):
                            org[cur_ll,labels[cur_ll]] = 1 - label_smoothing
                       labels = org.cuda()
                       #print labels.size()

                    # forward
                    outputs = model(inputs)
                    if isinstance(outputs, list) or isinstance(outputs, tuple): ##for inception_v3
                        loss = criterion(outputs[0], labels)
                        loss += criterion(outputs[1], labels)
                        outputs = outputs[0]

                    else:
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # statistics
                    val_loss += loss.data.item()
                    if label_smoothing > 0.0:
                        _ , true_label = torch.max(labels,1)
                        batch_corrects = torch.sum((preds == true_label)).item()
                    else: 
                        batch_corrects = torch.sum((preds == labels)).item()
                    val_corrects += batch_corrects

                val_loss = val_loss / val_size
                val_acc = 1.0 * val_corrects / len(data_set['val'])

                t1 = time.time()
                since = t1-t0
                logging.info('--'*30)
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())

                logging.info('%s epoch[%d]-val-loss: %.4f ||val-acc@1: %.4f ||time: %d'
                             % (dt(), epoch, val_loss, val_acc, since))
                
                if val_acc < 0.5 or  val_best_acc > val_acc:
                    continue
                val_best_acc = val_acc 
                # save model
                save_path = os.path.join(save_dir,
                        '%s-weights-%d-%d-[%.4f].pth'%(model_name,epoch,batch_cnt,val_acc))
                torch.save(model.state_dict(), save_path)
                logging.info('saved model to %s' % (save_path))
                logging.info('--' * 30)


