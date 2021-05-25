import torch
import scipy.io as sio
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def create_data_loader(feat_path, classes, batch_size = 100):
    feats = sio.loadmat(feat_path)
    labels = feats['labels']
    feats = feats['features']
    feats = np.transpose(feats,(1,0))

    train_data, test_data, train_label, test_label = train_test_split(feats, labels, test_size = 0.25, shuffle = False)

    train_label = torch.tensor(np.squeeze(train_label, axis = 1))
    test_label = torch.tensor(np.squeeze(test_label, axis = 1))

    data = []
    for i in range(len(train_label)):
        if (train_label[i] in classes):
            data.append([train_data[i,:], train_label[i]])

    data_test = []
    for i in range(len(test_label)):
        if (test_label[i] in classes):
            data_test.append([test_data[i,:], test_label[i]])

    train_loader = DataLoader(data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader