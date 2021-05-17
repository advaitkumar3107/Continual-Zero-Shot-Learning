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
from nets import *
from video_data_loader import video_dataset


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

parser = argparse.ArgumentParser(description='Video action recogniton training')
parser.add_argument('--logfile_name', type=str, default="generator_w_with_sem:10",
                    help='file name for storing the log file')
parser.add_argument('--gpu', type=int, default=3,
                    help='GPU ID, start from 0')
parser.add_argument('--epochs', type = int, default = 100, help='number of epochs to train the model for')
parser.add_argument('--snapshot', type = int, default = 50, help = 'model is saved after these many epochs')
parser.add_argument('--total_classes', type = int, default = 40, help = 'total number of classes to train over')
parser.add_argument('--test_interval', type = int, default = 1, help = 'number of epochs after which to test the model')
parser.add_argument('--train', type = int, default = 1, help = '1 if training. 0 for testing')
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
input_dim = 2048
nEpochs = args.epochs  # Number of epochs for training
useTest = True # See evolution of the test set when training
nTestInterval = args.test_interval # Run on test set every nTestInterval epochs
snapshot = args.snapshot # Store a model every snapshot epochs
lr = 1e-3 # Learning rate

dataset = 'ucf101' # Options: hmdb51 or ucf101

total_classes = args.total_classes

all_classes = np.arange(total_classes)

n_cl_temp = 0
class_map = {}
map_reverse = {}
for i, cl in enumerate(all_classes):
	if cl not in class_map:
		class_map[cl] = int(n_cl_temp)
		n_cl_temp += 1

print ("Class map:", class_map)

for cl, map_cl in class_map.items():
	map_reverse[map_cl] = int(cl)

print ("Map Reverse:", map_reverse)

print ("all_classes:", all_classes)


current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
save_dir_root = current_dir
save_dir = os.path.join(save_dir_root, 'run', log_name)
modelName = 'Bi-LSTM' # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return Variable(outputs.data, requires_grad=True).cuda()

def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')



def train_model(dataset=dataset, save_dir=save_dir, num_classes=total_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    model = ConvLSTM(
        num_classes=num_classes,
        latent_dim=512,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )
    classifier = Classifier(num_classes = num_classes)

    print("Training {} from scratch...".format(modelName))

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    log_dir = os.path.join(save_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))

    train_dataset = video_dataset(train = True, classes = all_classes[:num_classes])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = video_dataset(train = False, classes = all_classes[:num_classes])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'test': test_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'test']}
    lab_list = []
    pred_list = []

        # att = np.load("../npy_files/seen_semantic_51.npy")
        # att = torch.tensor(att).cuda()    

    if cuda:
        model = model.to(device)
        classifier = classifier.to(device)

    optimizer = torch.optim.Adam(list(model.parameters())+list(classifier.parameters()), lr=lr)

    if args.train == 1:
            
        for epoch in range(num_epochs):
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            model.train()
            classifier.train()

            for indices, inputs, labels in (trainval_loaders["train"]):
                inputs = inputs.permute(0,2,1,3,4)
                image_sequences = Variable(inputs.to(device), requires_grad=True)
                labels = Variable(labels.to(device), requires_grad=False)                

                model.lstm.reset_hidden_state()
                loop_batch_size = len(inputs)

                true_features_2048 = model(image_sequences)
                true_features_2048 = true_features_2048.view(true_features_2048.size(0), -1)
                logits = classifier(true_features_2048)

                loss = nn.CrossEntropyLoss()(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predictions = torch.max(torch.softmax(logits, dim = 1), dim = 1, keepdim = False)
                                  
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            epoch_loss = running_loss / trainval_sizes["train"]
            epoch_acc = running_corrects.double() / trainval_sizes["train"]

            writer.add_scalar('data/train_acc_epoch_benchmark {}'.format(i), epoch_acc, epoch)
            print("[test] Epoch: {}/{} Training Acc: {}".format(epoch+1, num_epochs, epoch_acc))


            if useTest and epoch % test_interval == (test_interval - 1):
                model.eval()
                classifier.eval()
                start_time = timeit.default_timer()

                running_loss = 0.0
                running_corrects = 0.0

                for indices, inputs, labels in (trainval_loaders["test"]):
                    inputs = inputs.permute(0,2,1,3,4)
                    image_sequences = Variable(inputs.to(device), requires_grad=False)
                    labels = Variable(labels.to(device), requires_grad=False)              
                    loop_batch_size = len(inputs)

                    with torch.no_grad():
                        model.lstm.reset_hidden_state()
                        true_features_2048 = model(image_sequences)
                        probs = classifier(true_features_2048)


                    _, predictions = torch.max(torch.softmax(probs, dim = 1), dim = 1, keepdim = False)
                    running_corrects += torch.sum(predictions == labels.data)

                real_epoch_acc = running_corrects.double() / trainval_sizes["test"]

                writer.add_scalar('data/test_acc_epoch_benchmark {}'.format(i), real_epoch_acc, epoch)

                print("[test] Epoch: {}/{} Test Dataset Acc: {}".format(epoch+1, num_epochs, real_epoch_acc))
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")


            if (epoch % save_epoch == save_epoch - 1):
                save_path = os.path.join(save_dir, saveName + '_increment' '_epoch-' + str(epoch) + '.pth.tar')
                torch.save({
                        'epoch': epoch + 1,
                        'extractor_state_dict': model.state_dict(),
                        'classifier_state_dict': classifier.state_dict(),
                        'num_classes': num_classes,
                    }, save_path)
                print("Save model at {}\n".format(save_path))
        
    writer.close()

if __name__ == "__main__":
    train_model()
    print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")    