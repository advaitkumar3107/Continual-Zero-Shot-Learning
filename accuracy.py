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
parser.add_argument('--logfile_name', type=str, default="generator_w_with_sem:10",
                    help='file name for storing the log file')
parser.add_argument('--load_name', type = str, default = "temp", help = 'Name of the directory which contains the saved weights')
parser.add_argument('--gpu', type=int, default=3,
                    help='GPU ID, start from 0')
parser.add_argument('--start_class', type = int, default = 0, help = 'testing classes start from')
parser.add_argument('--end_class', type = int, default = 20, help = 'testing classes end at')
parser.add_argument('--resume_epoch', type = int, default = None, help = 'Epoch from where to load weights')
parser.add_argument('--feat_path', type = str, default = "ucf101_i3d/i3d.mat", help = 'Path which contains the pretrained feats')
parser.add_argument('--att_path', type = str, default = "ucf101_i3d/split_1/att_splits.mat", help = 'Path which contains the pretrained attributes')

args = parser.parse_args()

gpu_id = str(args.gpu)
log_name = args.logfile_name
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device, "| gpu_id: ", gpu_id)
std_start_time = time.time()

batch_size = 100
semantic_dim = 300
input_dim = 8192
noise_dim = 1024

dataset = 'ucf101' # Options: hmdb51 or ucf101

num_classes = args.end_class
all_classes = range(num_classes)
test_classes = all_classes[args.start_class:]

current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
save_dir_root = current_dir
save_dir = os.path.join(save_dir_root, 'run', log_name)
load_dir = os.path.join(save_dir_root, 'run', args.load_name)
modelName = 'Bi-LSTM' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

att_path = args.att_path
feat_path = args.feat_path

def test_model(dataset=dataset, save_dir=save_dir, load_dir = load_dir):

    generator = Modified_Generator(semantic_dim, noise_dim)
    discriminator = Discriminator(input_dim = input_dim)
    classifier = Classifier(num_classes = num_classes)

    checkpoint = torch.load(os.path.join(load_dir, saveName + '_increment' '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
    print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])    
    print("Training {} saved model...".format(modelName))

    print('Total params: %.2fM' % (sum(p.numel() for p in classifier.parameters()) / 1000000.0))

    log_dir = os.path.join(save_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))

    train_dataloader, test_dataloader, len_train, len_test = create_data_loader(feat_path, all_classes)

    trainval_loaders = {'train': train_dataloader, 'test': test_dataloader}

    if cuda:
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        classifier = classifier.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    optimizer = torch.optim.Adam(list(classifier.parameters()), lr=lr)

    adversarial_loss = torch.nn.BCELoss().to(device)    

    feats = sio.loadmat(att_path)
    att = feats['att']
    att = np.transpose(att, (1,0))
    att = torch.tensor(att).cuda()  

      
    if args.train == 1:
        if args.only_gan == 0:    
            for epoch in range(num_epochs):
                start_time = timeit.default_timer()

                running_loss = 0.0
                running_corrects = 0.0

                classifier.train()

                for (inputs, labels) in (trainval_loaders["train"]):
                    feats = Variable(inputs, requires_grad = True).float().cuda()
                    labels = Variable(labels, requires_grad = False).long().cuda()

                    optimizer.zero_grad()

                    #model.lstm.reset_hidden_state()
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
                    #classifier.eval()
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
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'classifier_state_dict': classifier.state_dict(),
                        }, save_path)
                    print("Save model at {}\n".format(save_path))

if __name__ == "__main__":
    test_model()
    print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")    