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
from dataloader import create_data_loader, create_old_data_loader, create_data_loader_zsl
import scipy.io as sio
from models.nets import *

train_loader, test_loader, len_train, len_test = create_data_loader_zsl("gen_features/gzsl_features/gen_feat_labs_101_0.npy")

leng = 0

for (inputs, labels) in train_loader:
    print(labels)