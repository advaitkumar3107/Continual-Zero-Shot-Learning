from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
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
from torch.utils.data import DataLoader
from torch.autograd import Variable
import requests
from models.nets import *
from video_data_loader import video_dataset
from model import generate_model, load_pretrained_model
from dataloader import create_data_loader
import scipy.io as sio


parser = argparse.ArgumentParser(description='Video action recogniton testing')

parser.add_argument('--load_name', type = str, default = "temp", help = 'Name of the directory which contains the saved weights')
parser.add_argument('--gpu', type=int, default=3,
                    help='GPU ID, start from 0')
parser.add_argument('--start_class', type = int, default = 0, help = 'testing classes start from')
parser.add_argument('--end_class', type = int, default = 20, help = 'testing classes end at')
parser.add_argument('--num_class', type = int, default = 20, help = 'Number of classes in classifier')
parser.add_argument('--resume_epoch', type = int, default = None, help = 'Epoch from where to load weights')
parser.add_argument('--feat_path', type = str, default = "ucf101_i3d/i3d.mat", help = 'Path which contains the pretrained feats')
parser.add_argument('--att_path', type = str, default = "ucf101_i3d/split_1/att_splits.mat", help = 'Path which contains the pretrained attributes')
parser.add_argument('--increment', type = int, default = None, help = 'Number of increments the model was trained for')
parser.add_argument('--dataset', type = str, default = "ucf101", help = 'Dataset to test on')
parser.add_argument('--only_classifier', type = int, default = 0, help = '1 if only classifier needs to be tested')

args = parser.parse_args()

gpu_id = str(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device, "| gpu_id: ", gpu_id)
std_start_time = time.time()

batch_size = 100
semantic_dim = 300
input_dim = 8192
noise_dim = 1024
resume_epoch = args.resume_epoch
increment = args.increment

dataset = args.dataset # Options: hmdb51 or ucf101

num_class = args.num_class
end_class = args.end_class
all_classes = range(end_class)
test_classes = all_classes[args.start_class:]

current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
save_dir_root = current_dir
load_dir = os.path.join(save_dir_root, 'run', args.load_name)
modelName = 'Bi-LSTM' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

att_path = args.att_path
feat_path = args.feat_path
only_classifier = args.only_classifier

def test_model(dataset=dataset, load_dir = load_dir, only_classifier = only_classifier):

    if (only_classifier == 0):
        generator = Modified_Generator(semantic_dim, noise_dim)
    
        if (increment is None):
            checkpoint = torch.load(os.path.join(load_dir, saveName + '_increment' +  '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                           map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
            print("Initializing weights from: {}...".format(
                os.path.join(load_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
            classifier = Classifier(num_classes = num_class, bias = True)

        else:
            checkpoint = torch.load(os.path.join(load_dir, saveName + '_increment_' + str(increment) + '_epoch-' + 'best' + '.pth.tar'),
                           map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
            print("Initializing weights from: {}...".format(
                os.path.join(load_dir, 'models', saveName + '_epoch-' + 'best' + '.pth.tar')))
            classifier = Classifier(num_classes = num_class, bias = False)

        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])    
        print("Training {} saved model...".format(modelName))

        print('Total params: %.2fM' % (sum(p.numel() for p in classifier.parameters()) / 1000000.0))

        print('Training model on {} dataset...'.format(dataset))

        _, test_dataloader, _, len_test = create_data_loader(feat_path, test_classes)
 
        if cuda:
            generator = generator.to(device)
            classifier = classifier.to(device)

        classifier.train()
        generator.train()

        feats = sio.loadmat(att_path)
        att = feats['att']
        att = np.transpose(att, (1,0))
        att = torch.tensor(att).cuda()  
      
        start_time = timeit.default_timer()

        running_corrects = 0.0

        for (inputs, labels) in test_dataloader:
            feats = Variable(inputs, requires_grad = False).float().cuda()
            labels = Variable(labels, requires_grad = False).long().cuda()
            loop_batch_size = len(feats)
            probs = classifier(feats)                
            _, predictions = torch.max(torch.softmax(probs, dim = 1), dim = 1, keepdim = False)

            running_corrects += torch.sum(predictions == labels.data)

        epoch_acc = running_corrects.item()/len_test
        print("[test] Classifier Testing Acc: {}".format(epoch_acc))



        start_time = timeit.default_timer()
        running_corrects = 0.0

        for (inputs, labels) in test_dataloader:
            feats = Variable(inputs, requires_grad = True).float().cuda()
            labels = Variable(labels, requires_grad = False).long().cuda()
            loop_batch_size = len(feats)
        
            noise = Variable(FloatTensor(np.random.normal(0, 1, (loop_batch_size, noise_dim))))
            semantic_true = att[labels]

            with torch.no_grad():
                features_2048 = generator(semantic_true.float(), noise)
                probs = classifier(features_2048)

            _, predictions = torch.max(torch.softmax(probs, dim = 1), dim = 1, keepdim = False)
            running_corrects += torch.sum(predictions == labels.data)

        real_epoch_acc = running_corrects.item() / len_test

        print("[test] Epoch: Test Dataset Generator Acc: {}".format(real_epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

    else:
        if (increment is None):
            checkpoint = torch.load(os.path.join(load_dir, saveName + '_increment' +  '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                           map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
            print("Initializing weights from: {}...".format(
                os.path.join(load_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
            classifier = Classifier(num_classes = num_class, bias = False)

        else:
            checkpoint = torch.load(os.path.join(load_dir, saveName + '_increment_' + str(increment) + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                           map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
            print("Initializing weights from: {}...".format(
                os.path.join(load_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
            classifier = Classifier(num_classes = num_class, bias = False)

        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        print("Training {} saved model...".format(modelName))

        print('Total params: %.2fM' % (sum(p.numel() for p in classifier.parameters()) / 1000000.0))

        print('Training model on {} dataset...'.format(dataset))

        _, test_dataloader, _, len_test = create_data_loader(feat_path, test_classes)
 
        if cuda:
            classifier = classifier.to(device)

        classifier.train()
        
        feats = sio.loadmat(att_path)
        att = feats['att']
        att = np.transpose(att, (1,0))
        att = torch.tensor(att).cuda()  
      
        start_time = timeit.default_timer()

        running_corrects = 0.0

        for (inputs, labels) in test_dataloader:
            feats = Variable(inputs, requires_grad = False).float().cuda()
            labels = Variable(labels, requires_grad = False).long().cuda()
            loop_batch_size = len(feats)
            probs = classifier(feats)                
            _, predictions = torch.max(torch.softmax(probs, dim = 1), dim = 1, keepdim = False)

            running_corrects += torch.sum(predictions == labels.data)

        epoch_acc = running_corrects.item()/len_test
        print("[test] Classifier Testing Acc: {}".format(epoch_acc))


if __name__ == "__main__":
    test_model()
    print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")    