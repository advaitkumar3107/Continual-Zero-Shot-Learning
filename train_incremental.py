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
from video_data_loader import video_dataset, old_video_dataset


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

parser = argparse.ArgumentParser(description='Video action recogniton training')
parser.add_argument('--logfile_name', type=str, default="generator_w_with_sem:10",
                    help='file name for storing the log file')
parser.add_argument('--load_name', type = str, default = None, help = 'file name for loading the log file')
parser.add_argument('--gpu', type=int, default=3,
                    help='GPU ID, start from 0')
parser.add_argument('--epochs', type = int, default = 100, help='number of epochs to train the model for')
parser.add_argument('--increment_epochs', type = int, default = 25, help = 'number of epochs to increase after each set')
parser.add_argument('--snapshot', type = int, default = 50, help = 'model is saved after these many epochs')
parser.add_argument('--resume_epoch', type = int, default = None, help = 'resume training from this epoch')
parser.add_argument('--num_classes', type = int, default = 40, help = 'number of classes in the pretrained classifier')
parser.add_argument('--total_classes', type = int, default = 70, help = 'total number of classes to train over')
parser.add_argument('--incremental_classes', type = int, default = 10, help = 'number of classes to add at each increment')
parser.add_argument('--test_interval', type = int, default = 1, help = 'number of epochs after which to test the model')
parser.add_argument('--train', type = int, default = 1, help = '1 if training. 0 for testing')
parser.add_argument('--importance', type = int, default = 100, help = 'multiplying factor for distillation loss')
parser.add_argument('--distillation', type = bool, default = True, help = 'whether to use distillation or not')
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
increment = args.increment_epochs
nEpochs = args.epochs - increment  # Number of epochs for training
resume_epoch = args.resume_epoch  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = args.test_interval # Run on test set every nTestInterval epochs
snapshot = args.snapshot # Store a model every snapshot epochs
lr = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5] # Learning rate

dataset = 'ucf101' # Options: hmdb51 or ucf101

num_classes = args.num_classes
total_classes = args.total_classes
increment_classes = args.incremental_classes

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
load_dir = os.path.join(save_dir_root, 'run', args.load_name)
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



def train_model(dataset=dataset, save_dir=save_dir, load_dir = load_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, increment = increment, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
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

    if args.resume_epoch is not None and args.train == 1:
        checkpoint = torch.load(os.path.join(load_dir, saveName + '_increment' '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['extractor_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
    elif args.resume_epoch is not None and args.train == 0:
        checkpoint = torch.load(os.path.join(load_dir, saveName + '_increment' + '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        
        model.load_state_dict(checkpoint['extractor_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])

    else:
        print("Training {} from scratch...".format(modelName))

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    log_dir = os.path.join(save_dir)
    writer = SummaryWriter(log_dir=log_dir)

    iters = (total_classes - num_classes)/(increment_classes)

    for i in range(int(iters)):

        print('Training model on {} dataset...'.format(dataset))
       
        model1 = deepcopy(model)
        classifier1 = deepcopy(classifier)
        model1, classifier1 = model1.cuda(), classifier1.cuda()
        print('Copied previous model')

        in_features = classifier.classifier_out.in_features
        weights = classifier.classifier_out.weight.data
        print('new out features: ', num_classes + increment_classes)
        classifier.classifier_out = nn.Linear(in_features, num_classes + increment_classes, bias = False)
        kaiming_normal_init(classifier.classifier_out.weight)
        classifier.classifier_out.weight.data[:num_classes] = weights
            #model1.eval()
            #classifier1.eval()
        print('Updated Classifier With Number Of Classes %d' % (num_classes + increment_classes))
            
        train_params = list(classifier.parameters())

        train_dataset = video_dataset(train = True, classes = all_classes[num_classes:increment_classes + num_classes])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset = video_dataset(train = False, classes = all_classes[num_classes:increment_classes + num_classes])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        print('Classes used in the new dataset: %d to %d' % (num_classes, num_classes+increment_classes))

        old_train_dataset = old_video_dataset(train = True, classes = all_classes[:num_classes], num_classes = num_classes, samples = 10)
        old_train_dataloader = DataLoader(old_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        old_test_dataset = video_dataset(train = False, classes = all_classes[:num_classes])
        old_test_dataloader = DataLoader(old_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        print('Classes used in the old dataset: 0 to %d' % (num_classes))

        trainval_loaders = {'train': train_dataloader, 'val': test_dataloader}
        trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
        test_size = len(test_dataloader.dataset)
        lab_list = []
        pred_list = []

        # att = np.load("../npy_files/seen_semantic_51.npy")
        # att = torch.tensor(att).cuda()    

        if cuda:
            model = model.to(device)
            classifier = classifier.to(device)

        num_epochs = num_epochs + increment
        num_lr_stages = num_epochs/len(lr)

        if args.train == 1:
            
            for epoch in range(num_epochs):
                start_time = timeit.default_timer()

                running_loss = 0.0
                running_corrects = 0.0

                optimizer = torch.optim.Adam(list(model.parameters())+list(classifier.parameters()), lr=lr[int(epoch/num_lr_stages)])
                optimizer1 = torch.optim.Adam(train_params, lr=lr[int(epoch/num_lr_stages)])

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

                    cls_loss = nn.CrossEntropyLoss()(logits, labels)

                    if (args.distillation):
                        model1.lstm.reset_hidden_state()
                        dist_target = model1(image_sequences)
                        dist_target = classifier1(dist_target)
                        logits_dist = logits[:, :num_classes]
                        dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 0.5)
                        #dist_loss = nn.MSELoss()(logits1, dist_target.detach())
                        loss = args.importance*dist_loss + cls_loss
                        optimizer1.zero_grad()
                        loss.backward()
                        optimizer1.step()

                    else:
                        loss = cls_loss
                        optimizer1.zero_grad()
                        loss.backward()
                        optimizer1.step()

                    _, predictions = torch.max(torch.softmax(logits, dim = 1), dim = 1, keepdim = False)
                                  
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predictions == labels.data)

                epoch_loss = running_loss / trainval_sizes["train"]
                epoch_acc = running_corrects.double() / trainval_sizes["train"]

                words = 'data/train_acc_epoch' + str(i)
                writer.add_scalar(words, epoch_acc, epoch)
                print("Set: {} Epoch: {}/{} Training Acc: {}".format(i, epoch+1, num_epochs, epoch_acc))



                running_loss = 0.0
                running_corrects = 0.0

                model.train()
                classifier.train()

                for k, (indices, inputs, labels) in enumerate(old_train_dataloader):
                    #if (k > 2):
                        #break

                    inputs = inputs.permute(0,2,1,3,4)
                    image_sequences = Variable(inputs.to(device), requires_grad=True)
                    labels = Variable(labels.to(device), requires_grad=False)                

                    model.lstm.reset_hidden_state()
                    loop_batch_size = len(inputs)

                    true_features_2048 = model(image_sequences)
                    true_features_2048 = true_features_2048.view(true_features_2048.size(0), -1)
                    logits = classifier(true_features_2048)

                    cls_loss = nn.CrossEntropyLoss()(logits, labels)
                    loss = cls_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    _, predictions = torch.max(torch.softmax(logits, dim = 1), dim = 1, keepdim = False)
                                  
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predictions == labels.data)

                epoch_loss = running_loss / (batch_size * 2)
                epoch_acc = running_corrects.double() / (batch_size * 2)

                words = 'data/old_train_acc_epoch' + str(i)
                writer.add_scalar(words, epoch_acc, epoch)

                print("Set: {} Epoch: {}/{} Old Dataset Training Acc: {}".format(i, epoch+1, num_epochs, epoch_acc))



                if useTest and epoch % test_interval == (test_interval - 1):
                    model.eval()
                    classifier.eval()
                    start_time = timeit.default_timer()

                    running_loss = 0.0
                    running_corrects = 0.0

                    for indices, inputs, labels in test_dataloader:
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

                    real_epoch_acc = running_corrects.double() / test_size

                    words = 'data/test_acc_epoch' + str(i)
                    writer.add_scalar(words, real_epoch_acc, epoch)

                    print("Set: {} Epoch: {}/{} Test Dataset Acc: {}".format(i, epoch+1, num_epochs, real_epoch_acc))
                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time) + "\n")

                    if (args.distillation):
                        running_loss = 0.0
                        running_corrects = 0.0

                        for indices, inputs, labels in old_test_dataloader:
                            inputs = inputs.permute(0,2,1,3,4)
                            image_sequences = Variable(inputs.to(device), requires_grad=False)
                            labels = Variable(labels.to(device), requires_grad=False)              
                            loop_batch_size = len(inputs)

                            with torch.no_grad():
                                model.lstm.reset_hidden_state()
                                true_features_2048 = model(image_sequences)
                                probs = classifier(true_features_2048)

                            _, predictions = torch.max(torch.softmax(probs, dim = 1), dim = 1, keepdim = False)
                            print(predictions)
                            running_corrects += torch.sum(predictions == labels.data)

                        real_epoch_acc = running_corrects.double() / len(old_test_dataloader.dataset)

                        words = 'data/old_test_acc_epoch' + str(i)
                        writer.add_scalar(words, real_epoch_acc, epoch)

                        print("Set: {} Epoch: {}/{} Old Dataset Acc: {}".format(i, epoch+1, num_epochs, real_epoch_acc))

                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time) + "\n")         


                if (epoch % save_epoch == save_epoch - 1):
                    save_path = os.path.join(save_dir, saveName + '_increment_' + str(i) + '_epoch-' + str(epoch) + '.pth.tar')
                    torch.save({
                        'epoch': epoch + 1,
                        'extractor_state_dict': model.state_dict(),
                        'classifier_state_dict': classifier.state_dict(),
                        'num_classes': num_classes,
                    }, save_path)
                    print("Save model at {}\n".format(save_path))



        else:
            for epoch in range(num_epochs):
                start_time = timeit.default_timer()

                running_loss = 0.0
                running_corrects = 0.0

                model.eval()
                classifier.eval()

                for indices, inputs, labels in test_dataloader:
                    inputs = inputs.permute(0,2,1,3,4)
                    image_sequences = Variable(inputs.to(device), requires_grad=True)
                    labels = Variable(labels.to(device), requires_grad=False)                

                    model.lstm.reset_hidden_state()
                    loop_batch_size = len(inputs)

                    true_features_2048 = model(image_sequences)
                    true_features_2048 = true_features_2048.view(true_features_2048.size(0), -1)
                    logits = classifier(true_features_2048)

                    _, predictions = torch.max(torch.softmax(logits, dim = 1), dim = 1, keepdim = False)
                                  
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predictions == labels.data)

                epoch_loss = running_loss / len(test_dataloader)
                epoch_acc = running_corrects.double() / len(test_dataloader)

                writer.add_scalar('data/test_acc {}'.format(i), epoch_loss, epoch)
                print("[test] Epoch: {}/{} Testing Acc: {}".format(epoch+1, num_epochs, epoch_acc))


                running_loss = 0.0
                running_corrects = 0.0

                model.eval()
                classifier.eval()

                for indices, inputs, labels in old_dataloader:
                    inputs = inputs.permute(0,2,1,3,4)
                    image_sequences = Variable(inputs.to(device), requires_grad=True)
                    labels = Variable(labels.to(device), requires_grad=False)                

                    model.lstm.reset_hidden_state()
                    loop_batch_size = len(inputs)

                    true_features_2048 = model(image_sequences)
                    true_features_2048 = true_features_2048.view(true_features_2048.size(0), -1)
                    logits = classifier(true_features_2048)

                    _, predictions = torch.max(torch.softmax(logits, dim = 1), dim = 1, keepdim = False)
                                  
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predictions == labels.data)

                epoch_loss = running_loss / len(old_dataloader)
                epoch_acc = running_corrects.double() / len(old_dataloader)

                writer.add_scalar('data/old_acc {}'.format(i), epoch_loss, epoch)
                print("[test] Epoch: {}/{} Old Acc: {}".format(epoch+1, num_epochs, epoch_acc))
 
        num_classes = num_classes + increment_classes
        
    writer.close()

if __name__ == "__main__":
    train_model()
    print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")    