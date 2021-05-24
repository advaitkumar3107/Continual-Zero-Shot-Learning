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

from opts import parse_opts
from model import generate_model, load_pretrained_model
import scipy.io as sio

train_feats = np.load("convlstm_feat_labs_40.npy")
train_feats = torch.tensor(train_feats)

train_dataloader = DataLoader(train_feats, batch_size = 100, shuffle=True, num_workers=4)

for indices in train_dataloader:
    print(indices.shape)