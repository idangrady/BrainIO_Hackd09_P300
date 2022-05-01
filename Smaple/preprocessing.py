import scipy.io
import numpy as np
import pandas as pd
# import scipy as sc
# from scipy.signal import decimate # TODO see later if needed the decimation step!!
import random
np.random.seed(0)
random.seed(0)
import mat73

mat = mat73.loadmat('D:\github_\BrainIO_Hackd09_P300\DataPreprocessing\data_prepro.mat')['dat']

print(mat[0].keys())
# mat = scipy.io.loadmat('D:\github_\BrainIO_Hackd09_P300\DataPreprocessing\data_prepro.mat')['dat'][0]

def convertto3D(data, shape):
    return data.reshape(((shape)))

def dloader_2():
    participants = 5

    x , y = [],[]
    for par in range(participants):
        (_,n, m) = np.array(mat[par]['dat']).shape
        h = 28
        ar = np.array(mat[par]['dat']).reshape((-1, n*m))

        shape = -1, h , h,1  # len(mat[par]['trig'])
        x.append(convertto3D(ar[:,:h*h], shape)), y.append(mat[par]['trig'][:h*h])
    return x,y

def loadDataset(test_dataset_size = 500):
    dataset = {'sample_id':[],'labels':[], 'X':[]}
    timestamp = 20
    for subject in range(5):
        label, x = list(mat[subject]['trig']), (mat[subject]['dat'])
        for i in range(len(label)):
            if len(x[i]) >= timestamp: # drop short erroneous samples!
                dataset['labels'].append(label[i][0])
                dataset['X'].append(x[i][:timestamp]) # TODO change when real timestamp found!


    nb_true = 0
    for i in range(len(dataset['labels'])):
        if dataset['labels'][i] == 1:
            nb_true += 1

    # print(nb_true)

    dataset['sample_id'] = [i for i in range(len(dataset['labels'])) ]

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
    print(f"Shape True Signal {true_signal.shape}")
    # assert (true_signal.shape > 2)
    return true_signal




print("Import Preprocessing")