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
from dataloader import create_data_loader_zsl
import scipy.io as sio



def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

parser = argparse.ArgumentParser(description='Video action recogniton training from generated features')
parser.add_argument('--logfile_name', type=str, default="generator_w_with_sem:10",
                    help='file name for storing the log file')
parser.add_argument('--load_name', type = str, default = "temp", help = 'Name of the directory which contains the saved weights')
parser.add_argument('--gpu', type=int, default=3,
                    help='GPU ID, start from 0')
parser.add_argument('--epochs', type = int, default = 100, help='number of epochs to train the model for')
parser.add_argument('--snapshot', type = int, default = 50, help = 'model is saved after these many epochs')
parser.add_argument('--total_classes', type = int, default = 101, help = 'total number of classes to train over')
parser.add_argument('--test_interval', type = int, default = 1, help = 'number of epochs after which to test the model')
parser.add_argument('--train', type = int, default = 1, help = '1 if training. 0 for testing')
parser.add_argument('--resume_epoch', type = int, default = None, help = 'Epoch from where to load weights')
parser.add_argument('--feat_path', type = str, default = "gen_features/gzsl_features/gen_feat_labs_101_0.npy", help = 'Path which contains the generated feats')


args = parser.parse_args()

gpu_id = str(args.gpu)
log_name = args.logfile_name
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device, "| gpu_id: ", gpu_id)
std_start_time = time.time()

b1=0.5
b2=0.999
batch_size = 100
input_dim = 8192
semantic_dim = 300
noise_dim = 1024
nEpochs = args.epochs  # Number of epochs for training
resume_epoch = args.resume_epoch  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = args.test_interval # Run on test set every nTestInterval epochs
snapshot = args.snapshot # Store a model every snapshot epochs
lr = 5e-4 # Learning rate

dataset = 'ucf101' # Options: hmdb51 or ucf101

total_classes = args.total_classes

all_classes = range(total_classes)

current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
save_dir_root = current_dir
save_dir = os.path.join(save_dir_root, 'run', log_name)
load_dir = os.path.join(save_dir_root, 'run', args.load_name)
modelName = 'Bi-LSTM' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

feat_path = args.feat_path


def train_model(dataset=dataset, save_dir=save_dir, load_dir = load_dir, num_classes=total_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    classifier = Classifier(num_classes = num_classes, bias = False)

    if args.resume_epoch is not None:
        checkpoint = torch.load(os.path.join(load_dir, saveName + '_increment' '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
        print("Training {} saved model...".format(modelName))


    else:
        print("Training {} from scratch...".format(modelName))

    print('Total params: %.2fM' % (sum(p.numel() for p in classifier.parameters()) / 1000000.0))

    log_dir = os.path.join(save_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))

    train_dataloader, test_dataloader, len_train, len_test = create_data_loader_zsl(feat_path, all_classes)

    trainval_loaders = {'train': train_dataloader, 'test': test_dataloader}

    if cuda:
        classifier = classifier.to(device)

    optimizer = torch.optim.Adam(list(classifier.parameters()), lr=lr)
      
    if args.train == 1:
        
        for epoch in range(num_epochs):
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0
            classifier.train()

            for (inputs, labels) in (trainval_loaders["train"]):
                feats = Variable(inputs, requires_grad = True).float().cuda()
                labels = Variable(labels, requires_grad = False).long().cuda()

                optimizer.zero_grad()
                loop_batch_size = len(feats)

                probs = classifier(feats)
                loss = nn.CrossEntropyLoss()(probs, labels)
                loss.backward()
                optimizer.step()
                
                _, predictions = torch.max(torch.softmax(probs, dim = 1), dim = 1, keepdim = False)
                running_loss += loss.item() * feats.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            epoch_loss = running_loss/len_train
            epoch_acc = running_corrects.item()/len_train

            writer.add_scalar('data/train_acc_epoch_benchmark', epoch_acc, epoch)
            print("[train] Epoch: {}/{} Training Acc: {}".format(epoch+1, num_epochs, epoch_acc))


            if useTest and epoch % test_interval == (test_interval - 1):
                start_time = timeit.default_timer()

                running_corrects = 0.0

                for (inputs, labels) in (trainval_loaders["test"]):
                    feats = Variable(inputs, requires_grad = True).float().cuda()
                    labels = Variable(labels, requires_grad = False).long().cuda()
                    loop_batch_size = len(feats)

                    with torch.no_grad():
                        probs = classifier(feats)

                    _, predictions = torch.max(torch.softmax(probs, dim = 1), dim = 1, keepdim = False)
                    running_corrects += torch.sum(predictions == labels.data)

                    #print(len_test)

                real_epoch_acc = running_corrects.item()/len_test

                writer.add_scalar('data/test_acc_epoch_benchmark', real_epoch_acc, epoch)

                print("[test] Epoch: {}/{} Test Dataset Acc: {}".format(epoch+1, num_epochs, real_epoch_acc))
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")


            if (epoch % save_epoch == save_epoch - 1):
                save_path = os.path.join(save_dir, saveName + '_increment' '_epoch-' + str(epoch) + '.pth.tar')
                torch.save({
                    'classifier_state_dict': classifier.state_dict(),
                    }, save_path)
                print("Save model at {}\n".format(save_path))
    
    else:
        start_time = timeit.default_timer()

        running_corrects = 0.0

        for (inputs, labels) in (trainval_loaders["test"]):
            feats = Variable(inputs, requires_grad = True).float().cuda()
            labels = Variable(labels, requires_grad = False).long().cuda()
            loop_batch_size = len(feats)

            with torch.no_grad():
                probs = classifier(feats)

            _, predictions = torch.max(torch.softmax(probs, dim = 1), dim = 1, keepdim = False)
            running_corrects += torch.sum(predictions == labels.data)

                    #print(len_test)

        real_epoch_acc = running_corrects.item()/len_test

        writer.add_scalar('data/test_acc_epoch_benchmark', real_epoch_acc, epoch)

        print("[test] Epoch: {}/{} Test Dataset Acc: {}".format(epoch+1, num_epochs, real_epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")
            
    writer.close()

if __name__ == "__main__":
    train_model()
    print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")    