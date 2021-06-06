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
from models.nets import *

checkpoint = torch.load('run/pipeline_incremental/Bi-LSTM-ucf101_increment_1_epoch-499.pth.tar',map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU

state_dict = checkpoint['classifier_state_dict']

classifier = Classifier(num_classes = 50)
print(state_dict)