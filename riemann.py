import numpy as np
import matplotlib.pyplot as plt
from dataLoadInDataset import *
# from sklearn.model_selection import cross_val_score, RepeatedKFold

from pyriemann.utils.mean import mean_riemann
from pyriemann.classification import *
from pyriemann.estimation import *
from sklearn.covariance import ledoit_wolf_shrinkage, empirical_covariance # covariance estimation

inbalance_weight = 1.0 # weight to give to minority class
# inbalance_weight = 100.0 # weight to give to minority class


dataset, train_ids, test_ids, input_dims = loadDataset()

# ===== build numpy train array used for riemann method

train_data_list = []
train_labels_list = []
for id in train_ids:
    # train_data_list.append(dataset['X'][id])
    # train_data_list.append(ledoit_wolf_shrinkage(dataset['X'][id]))
    train_data_list.append(empirical_covariance(dataset['X'][id]))
    # train_data_list.append(XdawnCovariances(dataset['X'][id]))
    # train_data_list.append(Covariances(dataset['X'][id]))
    # train_data_list.append(dataset['X'][id])
    train_labels_list.append(dataset['labels'][id])
train_data = np.stack(train_data_list, axis=0)
# train_data = np.stack(train_data_list, axis=0).reshape(len(train_labels_list), 1)
train_labels = np.stack(train_labels_list, axis=0)#.reshape(len(train_labels_list), 1)
# print(train_data.shape)
# print(train_labels.shape)



# model= MDM(metric=dict(mean='riemann', distance='riemann')) #72%
model= TSclassifier(metric='riemann') #87.2%
# model= FgMDM(metric='riemann') #71.6


# train_data = train_data[:3]
# train_labels = np.array([0,1,0])
# print(train_data.shape)
# print(train_labels)

sample_weights = [1.0 for i in range(len(train_labels))]
for i in range(len(train_labels)):
    if train_labels[i] == 1:
        sample_weights[i] = inbalance_weight

model.fit(train_data, train_labels, np.array(sample_weights))



test_data_list = []
test_labels_list = []
for id in test_ids:
    # train_data_list.append(dataset['X'][id])
    # test_data_list.append(ledoit_wolf_shrinkage(dataset['X'][id]))
    test_data_list.append(empirical_covariance(dataset['X'][id]))
    # test_data_list.append(XdawnCovariances(dataset['X'][id]))
    # test_data_list.append(Covariances(dataset['X'][id]))
    # test_data_list.append(dataset['X'][id])
    test_labels_list.append(dataset['labels'][id])
test_data = np.stack(test_data_list, axis=0)
# test_data = np.stack(test_data_list, axis=0).reshape(len(test_labels_list), 1)
# print(test_data.shape)

test_output = model.predict(test_data)
# print(test_output)
score = 0
for i in range(len(test_labels_list)):
    if test_labels_list[i] == test_output[i]:
        score += 1
print("\n \n====== Final score with rieman training & inbalance weight: ", inbalance_weight," ==> ", 100*score/len(test_labels_list), "/100")