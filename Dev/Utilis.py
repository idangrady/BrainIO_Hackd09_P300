import numpy as np
import scipy.io
import glob


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

