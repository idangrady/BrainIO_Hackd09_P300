import scipy.io
import numpy as np
import pandas as pd
# import scipy as sc
# from scipy.signal import decimate # TODO see later if needed the decimation step!!
import random
np.random.seed(0)
random.seed(0)


mat = scipy.io.loadmat('DataPreprocessing/data.mat')['dat'][0]

print(mat.shape)

def loadDataset(test_dataset_size = 500):
    dataset = {'sample_id':[],'labels':[], 'X':[]}
    timestamp = 20
    for subject in range(5):
        label, x = list(mat[subject][0][0])[1], list(mat[subject][0][0])[3][0]
        for i in range(len(label)):
            if len(x[i]) >= timestamp: # drop short erroneous samples!
                dataset['labels'].append(label[i][0])
                dataset['X'].append(x[i][:timestamp]) # TODO change when real timestamp found!
            # print(label[i][0])
            # print(x[i])
            # print(x[i].shape)

    nb_true = 0
    for i in range(len(dataset['labels'])):
        if dataset['labels'][i] == 1:
            nb_true += 1

    # print(nb_true)

    dataset['sample_id'] = [i for i in range(len(dataset['labels'])) ]
    # print(np.count())
    # print(dataset['X'][0])
    # print(dataset['epoch_id'][:])
    # print(len(dataset['labels']))
    # print(len(dataset['X']))
    # print(dataset['X'][0])
    # print(dataset['X'])
    # print(dataset['labels'])

    # ==== train/test shuffle (by id)
    test_dataset_size = test_dataset_size
    test_ids = np.random.choice(dataset["sample_id"], test_dataset_size)
    train_ids = [id for id in dataset["sample_id"] if id not in test_ids]

    #===== display info
    input_dims = [timestamp,8] # TODO change !! (when timestamp available)
    print("Total dataset size: ", len(dataset['sample_id']), "test dataset size: ", test_dataset_size, "input dimensions: ", input_dims)
    print('dataset keys: ', dataset.keys())
    print("total number of targets: ", len([i for i in dataset['labels'] if i== 1]) ,"Number of targets in the testing set: ", len([dataset['labels'][i] for i in test_ids if dataset['labels'][i] == 1]) )

    return dataset, train_ids, test_ids, input_dims


def getSignalPerCandidate():
    dataset, train_ids, test_ids, input_dims= loadDataset()
    y = np.array(dataset['labels'])
    itemindex = np.where(y == 1)
    itemindex = [int(x) for x in itemindex[0]]
    true_signal = np.array([dataset['X'][x] for x in itemindex])

    assert (true_signal.shape > 2)
    return true_signal



#
# dataset, train_ids, test_ids, input_dims= loadDataset()
# print(dataset.keys())
# print(test_ids)
# print(input_dims)
# print(len(dataset['X']))
# y = np.array(dataset['labels'])
# itemindex = np.where(y==1)
# print(type(itemindex))
# print(itemindex[0])
# itemindex = [int(x) for x in itemindex[0]]
# print(len(itemindex))
# print(type(itemindex))
#
# true_signal = np.array([dataset['X'][x] for x in itemindex])
#
# print(true_signal.shape)