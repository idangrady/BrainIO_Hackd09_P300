import scipy.io
import numpy as np
import pandas as pd
mat = scipy.io.loadmat('tom_data/data.mat')['dat'][0]

print(mat.shape)

dataset = {'epoch_id':[],'labels':[], 'X':[]}

for subject in range(5):
    label, x = list(mat[subject][0][0])[1], list(mat[subject][0][0])[3][0]
    for i in range(len(label)):
        dataset['labels'].append(label[i][0])
        dataset['X'].append(x[i][:20]) # TODO change when real timestamp found!
        # print(label[i][0])
        # print(x[i])
        # print(x[i].shape)

dataset['epoch_id'] = [i for i in range(len(dataset['labels'])) ]
# print(dataset['epoch_id'][:])
# print(len(dataset['labels']))
# print(len(dataset['X']))
# print(dataset['X'][0])
# print(dataset['X'])
# print(dataset['labels'])