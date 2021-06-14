import torch
import scipy.io as sio
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_data_loader(feat_path, classes, batch_size = 100):
    feats = sio.loadmat(feat_path)
    labels = feats['labels'] - 1
    feats = feats['features']
    feats = np.transpose(feats,(1,0))
    labels = np.squeeze(labels, axis = 1)

    train_data, test_data, train_label, test_label = train_test_split(feats, labels, test_size = 0.25, random_state = 42)

    train_data = StandardScaler().fit_transform(train_data)
    test_data = StandardScaler().fit_transform(test_data)

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

    train_loader = DataLoader(data, batch_size = batch_size, shuffle = True, drop_last = True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle = False, drop_last = True)

    return train_loader, test_loader, len_train, len_test

def create_old_data_loader(feat_path, classes, batch_size = 100, num_samples = 10):
    feats = sio.loadmat(feat_path)
    labels = feats['labels'] - 1
    feats = feats['features']
    feats = np.transpose(feats,(1,0))
    labels = np.squeeze(labels, axis = 1)

    train_data, test_data, train_label, test_label = train_test_split(feats, labels, test_size = 0.25, random_state = 42)

    train_data = StandardScaler().fit_transform(train_data)
    test_data = StandardScaler().fit_transform(test_data)

    train_label = torch.tensor(train_label)
    test_label = torch.tensor(test_label)

    data = []
    len_train = 0
    len_test = 0

    num_classes = classes[-1]
    samples = torch.ones((num_classes+1))*num_samples
    
    for i in range(len(train_label)):
        if (train_label[i].item() in classes):
            if (samples[train_label[i].item()] > 0):
                data.append([train_data[i,:], train_label[i]])
                samples[train_label[i].item()] -= 1
                len_train += 1

    data_test = []
    for i in range(len(test_label)):
        if (test_label[i] in classes):
            data_test.append([test_data[i,:], test_label[i]])
            len_test += 1

    train_loader = DataLoader(data, batch_size = batch_size, shuffle = True, drop_last = True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle = False, drop_last = True)

    return train_loader, test_loader, len_train, len_test
    

def create_data_loader_zsl(feat_path, classes = range(101), batch_size = 100):
    feats = np.load(feat_path)
    labels = feats[:,-1]
    feats = feats[:,:8192]

    train_data, test_data, train_label, test_label = train_test_split(feats, labels, test_size = 0.25, random_state = 42)

    train_data = StandardScaler().fit_transform(train_data)
    test_data = StandardScaler().fit_transform(test_data)

    train_label = torch.tensor(train_label)
    test_label = torch.tensor(test_label)

    data = []
    len_train = 0
    len_test = 0

    num_classes = classes[-1]
    
    for i in range(len(train_label)):
        if (train_label[i].item() in classes):
            data.append([train_data[i,:], train_label[i]])
            len_train += 1

    data_test = []
    for i in range(len(test_label)):
        if (test_label[i] in classes):
            data_test.append([test_data[i,:], test_label[i]])
            len_test += 1

    train_loader = DataLoader(data, batch_size = batch_size, shuffle = True, drop_last = True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle = False, drop_last = True)

    return train_loader, test_loader, len_train, len_test
