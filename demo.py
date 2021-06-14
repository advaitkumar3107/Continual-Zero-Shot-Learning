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

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

generator = Modified_Generator(300, 1024)
checkpoint = torch.load('run/pipeline_set1/Bi-LSTM-ucf101_increment_epoch-99.pth.tar',
           map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
print("Initializing weights from: {}...".format(
       'run/pipeline_set1/Bi-LSTM-ucf101_increment_epoch-99.pth.tar'))

num_class = 40
noise_dim = 1024

classifier = Classifier(num_classes = num_class, bias = True)
classifier.load_state_dict(checkpoint['classifier_state_dict'])
generator.load_state_dict(checkpoint['generator_state_dict'])    
print("Training {} saved model...".format("UCF101"))

print('Total params: %.2fM' % (sum(p.numel() for p in classifier.parameters()) / 1000000.0))
 
if cuda:
    generator = generator.to(device)
    classifier = classifier.to(device)

classifier.eval()
generator.eval()

att_path = 'ucf101_i3d/split_1/att_splits.mat'
feat_path = 'ucf101_i3d/i3d.mat'
train_dataloader, _, len_train, len_test = create_data_loader(feat_path, range(num_class))

feats = sio.loadmat(att_path)
att = feats['att']
att = np.transpose(att, (1,0))
att = torch.tensor(att).cuda()  
      
start_time = timeit.default_timer()

running_corrects = 0.0

for (inputs, labels) in train_dataloader:
    feats = Variable(inputs, requires_grad = False).float().cuda()
    labels = Variable(labels, requires_grad = False).long().cuda()
    loop_batch_size = len(feats)
    probs = classifier(feats)                
    _, predictions_classifier = torch.max(torch.softmax(probs, dim = 1), dim = 1, keepdim = False)
                       
    noise = Variable(FloatTensor(np.random.normal(0, 1, (loop_batch_size, noise_dim))))
    semantic_true = att[labels]
    features_2048 = generator(semantic_true.float(), noise)
    probs = classifier(features_2048)
    _, predictions = torch.max(torch.softmax(probs, dim = 1), dim = 1, keepdim = False)

    running_corrects += torch.sum(predictions == predictions_classifier)

print("[test] Total Corrects: {} Total: {}".format(running_corrects, len_train))