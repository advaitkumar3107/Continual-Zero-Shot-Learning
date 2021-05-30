import torch
import scipy.io as sio
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def create_data_loader(feat_path, classes, batch_size = 100):
    feats = sio.loadmat(feat_path)
    labels = feats['labels'] - 1
    feats = feats['features']
    feats = np.transpose(feats,(1,0))
    labels = np.squeeze(labels, axis = 1)

    train_data, test_data, train_label, test_label = train_test_split(feats, labels, test_size = 0.25, random_state = 42)

    train_label = torch.tensor(train_label)
    test_label = torch.tensor(test_label)

    data = []
    len_train = 0
    len_test = 0
    for i in range(len(train_label)):
        if (train_label[i] in classes):
            data.append([train_data[i,:], train_label[i]])
            len_train += 1

    data_test = []
    for i in range(len(test_label)):
        if (test_label[i] in classes):
            data_test.append([test_data[i,:], test_label[i]])
            len_test += 1

    train_loader = DataLoader(data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader, len_train, len_test