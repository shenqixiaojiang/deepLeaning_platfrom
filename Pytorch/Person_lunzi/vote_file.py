import os
import pandas as pd
from glob import glob
import numpy as np

file_path = glob('result/result_*csv') #[:3]
print "There are ", len(file_path) , 'files......\n'
for index,cur_p in enumerate(file_path):
    print cur_p
    cur = np.array(pd.read_csv(cur_p,header=None))
    if index == 0:
        pred_shape = cur.shape
    #print cur.shape
    cur_l = np.argmax(cur,axis=1).reshape(cur.shape[0],1)
    #print cur_l.shape
    if index == 0:
        cv_pred = cur_l
    else:
        cv_pred = np.concatenate((cv_pred,cur_l),axis=1)
    #print cv_pred.shape

all_same = 0
submit = np.zeros(pred_shape)
for index,line in enumerate(cv_pred):
    max_index = np.argmax(np.bincount(line))
    if np.bincount(line)[max_index] > 0:
    #if np.bincount(line)[max_index] == len(file_path):
        submit[index][max_index] = 1
        all_same += 1
    else:
        submit[index][0] = 1
print submit.shape
print "all same: ", all_same, all_same * 1.0 / 4838.0
result = pd.DataFrame(submit.astype(np.int8))
result.to_csv('result/vote_' + str(len(file_path)) + 'files.csv',index=False,header=None)

### get all one 
'''
cv_pred = np.ones(4838,dtype=np.int8)
submit = np.zeros((4838,17),dtype=np.int8)
for index,line in enumerate(cv_pred):
    submit[index][0] = 1
print submit.shape
result = pd.DataFrame(submit)
result.to_csv('result/1111.csv',index=False,header=None)
'''
