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


model = generate_model()
model = load_pretrained_model(model, 'saved_weights/resnet_50.pth')

#model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)
model = nn.Sequential(*list(model.children())[:-1])
model = model.cuda()

inputs = torch.ones((1, 3, 64, 112, 112)).cuda()
outputs = model(inputs)

outputs = outputs.view(outputs.shape[0], -1)
print(outputs.shape)