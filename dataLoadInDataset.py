# import scipy.io
import numpy as np
import pandas as pd
# import scipy as sc
# from scipy.signal import decimate # TODO see later if needed the decimation step!! ===> Median step OK
import random
import mat73

np.random.seed(0)
random.seed(0)


# mat = scipy.io.loadmat('tom_data/data.mat')['dat'][0]
data_dict = mat73.loadmat('data/data_prepro.mat')['dat']#[1]
# print(data_dict)
# print(len(data_dict))
# print(data_dict[0].keys())
# print(data_dict[0]['trig'])
# print(len(data_dict[0]['trig']))
# print(data_dict[0]['dat'][0].shape)
# print(data_dict[0]['dat'][1].shape)
# print(data_dict[0]['dat'][11].shape)
# print(len(data_dict[0]['dat']))
# print(mat.shape)



def loadDataset(test_dataset_size = 500):
    dataset = {'sample_id':[],'labels':[], 'X':[]}
    verification_check = len( data_dict[0]['dat'][0]) # in theory 125 lines :
    for subject in range(5):
        labels, x = data_dict[0]['trig'], data_dict[0]['dat']
        # print(labels, x[0])
        # print(x[0])
        for i in range(len(labels)):
            if len(x[i]) >= verification_check: # drop short erroneous samples!
                # dataset['labels'].append(labels[i][0])
                dataset['labels'].append(labels[i])
                # print(len(x), len(x[0]))
                dataset['X'].append(x[i])
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
    input_dims = [verification_check,8] # TODO change !! (when timestamp available)
    print("Total dataset size: ", len(dataset['sample_id']), "test dataset size: ", test_dataset_size, "input dimensions: ", input_dims)
    print('dataset keys: ', dataset.keys())
    print("total number of targets: ", len([i for i in dataset['labels'] if i== 1]) ,"Number of targets in the testing set: ", len([dataset['labels'][i] for i in test_ids if dataset['labels'][i] == 1]) )

    return dataset, train_ids, test_ids, input_dims