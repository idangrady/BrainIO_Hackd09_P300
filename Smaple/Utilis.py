import numpy as np
import scipy.io
import glob
import torch
import uuid
import matplotlib.pyplot as plt
import os


UNIQUE_RUN_ID = str(uuid.uuid4())
NOISE_DIMENSION = 50
BATCH_SIZE = 10

def assembleDataSet(l):
    arr  = np.array((0))
    for idx, ar in enumerate(l):
        if(idx<1):
            arr = ar
        else:
            arr = np.concatenate((arr, ar), axis = 0)
    return arr

def generate_noise(number_of_images = 1, noise_dimension = NOISE_DIMENSION, device=None):
  """ Generate noise for number_of_images images, with a specific noise_dimension """
  return torch.randn(number_of_images, noise_dimension, device=device)


def get_positive(arr):
    return arr[arr[:,-1]!= 0]

def get_list_data():
    col =['y','trig']
    nps = []
    names = glob.glob("../Data/*.mat")
    for name in names:
        mat = scipy.io.loadmat(f"../Data/{name}")
        arr = mat[col[0]]
        for c in range(1, len(col)):
            arr = np.concatenate((arr,mat[col[c]]), axis = 1)
        nps.append(arr)
    arr = assembleDataSet(nps)

    return arr


def get_device():
    """ Retrieve device based on settings and availability. """
    return torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")


def make_directory_for_run():
    """ Make a directory for this training run. """
    print(f'Preparing training run {UNIQUE_RUN_ID}')
    if not os.path.exists('./runs'):
        os.mkdir('./runs')
    os.mkdir(f'./runs/{UNIQUE_RUN_ID}')


def generate_image(generator, epoch=0, batch=0, device=get_device()):
    """ Generate subplots with generated examples. """
    images = []
    noise = generate_noise(BATCH_SIZE, device=device)
    generator.eval()
    images = generator(noise)
    plt.figure(figsize=(10, 10))
    for i in range(16):
        # Get image
        image = images[i]
        # Convert image back onto CPU and reshape
        image = image.cpu().detach().numpy()
        image = np.reshape(image, (28, 28))
        # Plot
        plt.subplot(4, 4, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    if not os.path.exists(f'./runs/{UNIQUE_RUN_ID}/images'):
        os.mkdir(f'./runs/{UNIQUE_RUN_ID}/images')
    plt.savefig(f'./runs/{UNIQUE_RUN_ID}/images/epoch{epoch}_batch{batch}.jpg')


def save_models(generator, discriminator, epoch):
    """ Save models at specific point in time. """
    torch.save(generator.state_dict(), f'./runs/{UNIQUE_RUN_ID}/generator_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'./runs/{UNIQUE_RUN_ID}/discriminator_{epoch}.pth')


def print_training_progress(batch, generator_loss, discriminator_loss):
    """ Print training progress. """
    print('Losses after mini-batch %5d: generator %e, discriminator %e' %
          (batch, generator_loss, discriminator_loss))

#
# mat = scipy.io.loadmat('../DataPreprocessing/data.mat')['dat'][0]
#
# def loadDataset(test_dataset_size = 500):
#     dataset = {'sample_id':[],'labels':[], 'X':[]}
#     timestamp = 20
#     for subject in range(5):
#         label, x = list(mat[subject][0][0])[1], list(mat[subject][0][0])[3][0]
#         for i in range(len(label)):
#             if len(x[i]) >= timestamp: # drop short erroneous samples!
#                 dataset['labels'].append(label[i][0])
#                 dataset['X'].append(x[i][:timestamp]) # TODO change when real timestamp found!
#             # print(label[i][0])
#             # print(x[i])
#             # print(x[i].shape)
#
#     nb_true = 0
#     for i in range(len(dataset['labels'])):
#         if dataset['labels'][i] == 1:
#             nb_true += 1
#
#     # print(nb_true)
#
#     dataset['sample_id'] = [i for i in range(len(dataset['labels'])) ]
#     # print(np.count())
#     # print(dataset['X'][0])
#     # print(dataset['epoch_id'][:])
#     # print(len(dataset['labels']))
#     # print(len(dataset['X']))
#     # print(dataset['X'][0])
#     # print(dataset['X'])
#     # print(dataset['labels'])
#
#     # ==== train/test shuffle (by id)
#     test_dataset_size = test_dataset_size
#     test_ids = np.random.choice(dataset["sample_id"], test_dataset_size)
#     train_ids = [id for id in dataset["sample_id"] if id not in test_ids]
#
#     #===== display info
#     input_dims = [timestamp,8] # TODO change !! (when timestamp available)
#     print("Total dataset size: ", len(dataset['sample_id']), "test dataset size: ", test_dataset_size, "input dimensions: ", input_dims)
#     print('dataset keys: ', dataset.keys())
#     print("total number of targets: ", len([i for i in dataset['labels'] if i== 1]) ,"Number of targets in the testing set: ", len([dataset['labels'][i] for i in test_ids if dataset['labels'][i] == 1]) )
#
#     return dataset, train_ids, test_ids, input_dims

print("Import Utilis")