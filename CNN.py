
# from dataLoadInDataset import *
from dataLoadInDataset2 import *
from torch import nn
import torch
import torch.optim as optim
from tqdm import *
import matplotlib.pyplot as plt
from kornia.losses import focal

print("\n================== CNN P300 data classifier ====================\n")

# ======= fetch dataset
dataset, train_ids, test_ids, input_dims = loadDataset() # test size = 500 by default
# for torch, convert every numpy array by a torch tensor
# print(dataset['X'][0])
for i in (dataset['sample_id']):
    # dataset['X'][i] = torch.tensor(dataset['X'][i])
    # resize to [batch_size=1, nb_channels=1, input_dims[0], input_dims[1]) for CNN model input
    # print(torch.tensor(dataset['X'][i]).size())
    # dataset['X'][i] = (torch.tensor(dataset['X'][i]).resize(1, 1, input_dims[0], input_dims[1]))
    dataset['X'][i] = (torch.from_numpy(dataset['X'][i]).float().resize(1, 1, input_dims[0], input_dims[1]))
# print(dataset['X'][0])

# modify the labels format for the deep network model
for i in range(len(dataset['labels'])):
    val = dataset['labels'][i]
    if dataset['labels'][i] == -1 :
        val = 0
    else:
        val = 1
    # dataset['labels'][i] = torch.from_numpy(np.array([val])).float() # for crossentropy loss
    dataset['labels'][i] = torch.from_numpy(np.array([val]))#.float()

# print(dataset['labels'])


# Keep in memory test set
test_list = []
test_labels_list = []
for id in test_ids:
    test_list.append(dataset['X'][id])
    test_labels_list.append(dataset['labels'][id])
test_tensor_input = torch.stack(test_list).resize(len(test_ids), 1, input_dims[0], input_dims[1])
test_tensor_labels = torch.stack(test_labels_list).resize(len(test_ids), 1)


# ======= model creation

# TODO change
model = nn.Sequential(
    nn.Conv2d(1,1, 5), #input_nb_channels=1, output_nb_channels=1, kernel size
    nn.SELU(),
    nn.Conv2d(1,1,3),
    nn.SELU(),
    nn.Flatten(),
    # resampled 25 version
    nn.Linear(38, 2), # Output size of the previous layer, output size = 1 (probability)
    # all 125 version
    # nn.Linear(238, 2), # Output size of the previous layer, output size = 1 (probability)
    nn.Dropout(),
    nn.Softmax()
    # nn.Linear(238, 1), # Output size of the previous layer, output size = 1 (probability)
    # nn.Sigmoid() # proba
)

# x = dataset['X'][0]
# label = dataset['labels'][0]
# print( x.size())
# print(label)
# out = model(x)
# print(out.size())

# === init weights # TODO see what is best with CNN
# for layer in model:
#     if isinstance(layer, nn.Linear):
#         nn.init.xavier_uniform_(layer.weight)


#========== hyperparameters (optuna)
# lr, w_decay = 0.0001139448084577777, 2.472252240823343e-07 # TODO change
# gamma = 74.55715021234711

lr, w_decay = 1e-4, 0.0 # TODO change
gamma = 0.0
# ==== optimizer & loss function
# lr, w_decay = 1e-3, 0.0 # TODO change
# lr, w_decay = 1e-4, 0.0 # TODO change
# lr, w_decay = 1e-2, 0.0 # TODO change
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay= w_decay)
# Loss = nn.BCEWithLogitsLoss()


# ==== training loop
losses = []
scores = []
epochs = 10000
# epochs = 1000
# epochs = 200
minibatch_size = 32
pbar = tqdm(range(epochs))
for epoch in pbar:
    pbar.set_description(str(epoch))

    # 1) fetch minibatch
    minibatch_ids = np.random.choice(train_ids, size=minibatch_size)
    minibatch_list = []
    minibatch_labels_list = []
    for id in minibatch_ids:
        minibatch_list.append(dataset['X'][id])
        minibatch_labels_list.append(dataset['labels'][id])
    minibatch_tensor_input = torch.stack(minibatch_list).resize(minibatch_size, 1, input_dims[0], input_dims[1])
    minibatch_tensor_labels = torch.stack(minibatch_labels_list).resize(minibatch_size, 1)
    # print(minibatch_tensor_labels.size())
    # print(minibatch_tensor_labels)

    # 2) get outputs probabilities (target, non target)
    minibatch_tensor_output = model(minibatch_tensor_input)
    # print(minibatch_tensor_output.size())

    #3) compute loss function
    # print(minibatch_tensor_output.resize(minibatch_size), minibatch_tensor_labels.resize(minibatch_size))
    # print(minibatch_tensor_output, minibatch_tensor_labels)
    optimizer.zero_grad()

    #$$$
    # print("$$$$")
    # a= torch.randn(3,2,1)
    # # a= torch.randn(3,1,1) # bug
    # b= torch.empty(3,1, dtype = torch.long).random_(2)
    # print(a)
    # print(b)
    # # print(focal.focal_loss(a,b, alpha = 0.2))
    # print(focal.focal_loss(a,b, alpha = 0.2).mean())
    # #$$$

    # loss = Loss(minibatch_tensor_output, minibatch_tensor_labels)
    # print(loss)
    # print(minibatch_tensor_output.size(), minibatch_tensor_labels.size())
    # loss = focal.focal_loss(minibatch_tensor_output, minibatch_tensor_labels, alpha=3)
    # loss = focal.focal_loss(minibatch_tensor_output, minibatch_tensor_labels.resize(minibatch_size), alpha=3)
    # print(minibatch_tensor_labels)
    # loss = focal.focal_loss(minibatch_tensor_output, minibatch_tensor_labels.resize(minibatch_size), alpha=3)
    # gamma = 50.0#75.0#50.0 # too little#10.0 too little #1.0 # to little #100.0 # too much
    # loss = focal.focal_loss(minibatch_tensor_output, minibatch_tensor_labels.resize(minibatch_size), alpha=1.0, gamma=gamma).mean()
    loss = -focal.focal_loss(minibatch_tensor_output, minibatch_tensor_labels.resize(minibatch_size), alpha=1.0, gamma=gamma).mean()
    # print(loss)
    losses.append(loss.detach())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1) # avoid exploding gradients
    optimizer.step()


    # 4) check test score :
    score = 0
    with torch.no_grad() :
        test_output_probas = model(test_tensor_input)
        # print(test_output_probas.size())
        # print(test_output_probas.size())
        # print(test_tensor_labels.size())
        # print(test_output_probas[:10])
        # print(test_tensor_labels[:10])
        difference = torch.abs(test_output_probas - test_tensor_labels).tolist()
        for d in difference:
            if d[0] < 0.5 :
                score += 1
    scores.append(score/len(test_ids))




# ========== final score
score = 0
with torch.no_grad() :
    test_output_probas = model(test_tensor_input)
    # print(test_output_probas.size())
    # print(test_output_probas.size())
    # print(test_tensor_labels.size())
    print(test_output_probas[:10])
    print(test_tensor_labels[:10])
    difference = torch.abs(test_output_probas - test_tensor_labels).tolist()
    for d in difference:
        if d[0] < 0.5 :
            score += 1

# ========== Plots
plt.figure()
plt.subplot(2,1,1)
plt.title("Loss")
plt.xlabel('epoch')
plt.plot(losses)
plt.subplot(2,1,2)
plt.plot(scores)
plt.title("Accuracy")
plt.xlabel('epoch')
plt.show()