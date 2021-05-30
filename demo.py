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
from dataloader import create_data_loader

all_classes = range(10)
train_dataloader, test_dataloader, len_train, len_test = create_data_loader('ucf101_i3d/i3d.mat', all_classes[:10])

sum = 0

for (inputs, labels) in train_dataloader:
    sum += (labels == 0).sum()

print(sum)    