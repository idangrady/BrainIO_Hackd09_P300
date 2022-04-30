from mne.externals.pymatreader import read_mat

file = read_mat("data.mat")

data = file["dat"]

print(data)