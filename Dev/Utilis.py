import numpy as np
import scipy.io
import glob
import torch


def assembleDataSet(l):
    arr  = np.array((0))
    for idx, ar in enumerate(l):
        if(idx<1):
            arr = ar
        else:
            arr = np.concatenate((arr, ar), axis = 0)
    return arr


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
