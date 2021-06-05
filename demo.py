from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
from torchsummary import summary
import pdb
import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import time
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import requests
from video_data_loader import video_dataset
from dataloader import create_data_loader, create_old_data_loader
import scipy.io as sio


#classes = range(101)

#train_loader, test_loader, len_train, len_test = create_old_data_loader('ucf101_i3d/i3d.mat', classes[90:])

#for (inputs, labels) in test_loader:
    #print(inputs.shape)

model = nn.BatchNorm1d(100)
tens = torch.ones((1,100))
output = model(tens)
print(output.shape)