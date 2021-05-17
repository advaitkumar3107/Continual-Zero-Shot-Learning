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
semantic_dim = 300
noise_dim = 1024
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
        latent_dim=512,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )

    classifier = Classifier(num_classes = num_classes)
    generator = Modified_Generator(semantic_dim, noise_dim)
    discriminator = Discriminator(input_dim=input_dim)


    if args.resume_epoch is not None and args.train == 1:
        checkpoint = torch.load(os.path.join(load_dir, saveName + '_increment' '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['extractor_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
    elif args.resume_epoch is not None and args.train == 0:
        checkpoint = torch.load(os.path.join(load_dir, saveName + '_increment' + '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        
        model.load_state_dict(checkpoint['extractor_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    else:
        print("Training {} from scratch...".format(modelName))

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    log_dir = os.path.join(save_dir)
    writer = SummaryWriter(log_dir=log_dir)

    iters = (total_classes - num_classes)/(increment_classes)

    for i in range(int(iters)):

        print('Training model on {} dataset...'.format(dataset))
       
        if (i != 0):
            num_classes = num_classes + increment_classes

        #model1 = deepcopy(model)
        #classifier1 = deepcopy(classifier)
        generator1 = deepcopy(generator)
        #discriminator1 = deepcopy(discriminator)
        #model1, classifier1, generator1, discriminator1 = model1.cuda(), classifier1.cuda(), generator1.cuda(), discriminator1.cuda()
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
            
        train_dataset = video_dataset(train = True, classes = all_classes[num_classes:increment_classes + num_classes])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataset = video_dataset(train = False, classes = all_classes[num_classes:increment_classes + num_classes])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        print('Classes used in the new dataset: %d to %d' % (num_classes, num_classes+increment_classes))

        old_train_dataset = old_video_dataset(train = True, classes = all_classes[:num_classes], num_classes = num_classes, samples = 10)
        old_train_dataloader = DataLoader(old_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        #old_train_dataset = video_dataset(train = True, classes = all_classes[:num_classes])
        #old_train_dataloader = DataLoader(old_train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

        old_test_dataset = video_dataset(train = False, classes = all_classes[:num_classes])
        old_test_dataloader = DataLoader(old_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print('Classes used in the old dataset: 0 to %d' % (num_classes))

        trainval_loaders = {'train': train_dataloader, 'val': test_dataloader}
        trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
        test_size = len(test_dataloader.dataset)
        lab_list = []
        pred_list = []

        att = np.load("../npy_files/seen_semantic_51.npy")
        att = torch.tensor(att).cuda()    

        adversarial_loss = torch.nn.BCELoss().to(device)    

        if cuda:
            model = model.to(device)
            classifier = classifier.to(device)
            generator = generator.to(device)
            generator1 = generator1.to(device)
            discriminator = discriminator.to(device)

        num_epochs = num_epochs + increment
        num_lr_stages = num_epochs/len(lr)

        if args.train == 1:
            
            for epoch in range(num_epochs):
                start_time = timeit.default_timer()

                running_old_corrects = 0.0
                running_new_corrects = 0.0

                optimizer = torch.optim.Adam(list(model.parameters())+list(classifier.parameters()), lr=lr[int(epoch/num_lr_stages)])

                model.train()
                classifier.train()
                #generator.train()
                #discriminator.train()

                for indices, inputs, labels in (trainval_loaders["train"]):
                    inputs = inputs.permute(0,2,1,3,4)
                    image_sequences = Variable(inputs.to(device), requires_grad=True)
                    labels = Variable(labels.to(device), requires_grad=False)                

                    model.lstm.reset_hidden_state()
                    loop_batch_size = len(inputs)

############### Begin Incremental Training Of Conv-LSTM Model ############################
                    optimizer.zero_grad()
                    old_labels = Variable(LongTensor(np.random.randint(0, num_classes, loop_batch_size))).cuda()
                    old_semantic = att[old_labels].cuda()
                    noise = Variable(FloatTensor(np.random.normal(0, 1, (loop_batch_size, noise_dim)))).cuda()
                   
                    old_features = generator1(old_semantic.float(), noise)
                    new_features = model(image_sequences)
                    new_logits = classifier(new_features)
                    old_logits = classifier(old_features)

                    _, dataset_inputs, dataset_labels = next(iter(old_train_dataloader))
                    dataset_inputs = dataset_inputs.permute(0,2,1,3,4)
                    image_sequences = Variable(dataset_inputs.to(device), requires_grad=True)
                    dataset_labels = Variable(dataset_labels.to(device), requires_grad=False)                

                    model.lstm.reset_hidden_state()
                    loop_batch_size = len(dataset_inputs)

                    dataset_logits = classifier(model(image_sequences))
                    
                    loss = nn.CrossEntropyLoss()(new_logits, labels) + nn.CrossEntropyLoss()(old_logits, old_labels) + nn.CrossEntropyLoss()(dataset_logits, dataset_labels)
                    loss.backward()
                    optimizer.step()                    
    
                    _, old_predictions = torch.max(torch.softmax(old_logits, dim = 1), dim = 1, keepdim = False)
                    _, dataset_predictions = torch.max(torch.softmax(dataset_logits, dim = 1), dim = 1, keepdim = False)
                    _, new_predictions = torch.max(torch.softmax(new_logits, dim = 1), dim = 1, keepdim = False)         
                    running_old_corrects += torch.sum(old_predictions == old_labels.data) 
                    running_old_corrects += torch.sum(dataset_predictions == dataset_labels.data)
                    running_new_corrects += torch.sum(new_predictions == labels.data)

                old_epoch_acc = running_old_corrects.double() / (2*trainval_sizes['train'])
                new_epoch_acc = running_new_corrects.double() / trainval_sizes['train']

                words = 'data/old_train_acc_epoch' + str(i)
                writer.add_scalar(words, old_epoch_acc, epoch)
                words = 'data/new_train_acc_epoch' + str(i)
                writer.add_scalar(words, new_epoch_acc, epoch)

                print("Set: {} Epoch: {}/{} Train Old Acc: {} Train New Acc: {}".format(i, epoch+1, num_epochs, old_epoch_acc, new_epoch_acc))


                if useTest and epoch % test_interval == (test_interval - 1):
                    model.eval()
                    classifier.eval()
                    
                    running_old_corrects = 0.0
                    running_new_corrects = 0.0
    
                    for indices, inputs, labels in test_dataloader:
                        inputs = inputs.permute(0,2,1,3,4)
                        image_sequences = Variable(inputs.to(device), requires_grad=True)
                        labels = Variable(labels.to(device), requires_grad=False)                

                        model.lstm.reset_hidden_state()
                        loop_batch_size = len(inputs)
                   
                        new_features = model(image_sequences)
                        new_logits = classifier(new_features)
     
                        _, new_predictions = torch.max(torch.softmax(new_logits, dim = 1), dim = 1, keepdim = False)         
                        running_new_corrects += torch.sum(new_predictions == labels.data)

                    new_epoch_acc = running_new_corrects.double() / len(test_dataloader.dataset)

                    words = 'data/new_test_acc_epoch' + str(i)
                    writer.add_scalar(words, new_epoch_acc, epoch)


                    for indices, inputs, labels in old_test_dataloader:
                        inputs = inputs.permute(0,2,1,3,4)
                        image_sequences = Variable(inputs.to(device), requires_grad=True)
                        labels = Variable(labels.to(device), requires_grad=False)                

                        model.lstm.reset_hidden_state()
                        loop_batch_size = len(inputs)
                   
                        old_features = model(image_sequences)
                        old_logits = classifier(old_features)
     
                        _, old_predictions = torch.max(torch.softmax(old_logits, dim = 1), dim = 1, keepdim = False)         
                        running_old_corrects += torch.sum(old_predictions == labels.data)

                    old_epoch_acc = running_old_corrects.double() / len(old_test_dataloader.dataset)

                    words = 'data/old_test_acc_epoch' + str(i)
                    writer.add_scalar(words, old_epoch_acc, epoch)

                print("Set: {} Epoch: {}/{} Test Old Acc: {} Test New Acc: {}".format(i, epoch+1, num_epochs, old_epoch_acc, new_epoch_acc))



            for epoch in range(num_epochs):
                start_time = timeit.default_timer()

                running_old_corrects = 0.0
                running_new_corrects = 0.0

                optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr[0], betas=(b1, b2))
                optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr[0], betas=(b1, b2))

                model.train()
                classifier.train()
                generator1.eval()
                generator.train()
                discriminator.train()

                for indices, inputs, labels in (trainval_loaders["train"]):
                    inputs = inputs.permute(0,2,1,3,4)
                    image_sequences = Variable(inputs.to(device), requires_grad=True)
                    labels = Variable(labels.to(device), requires_grad=False)                

                    model.lstm.reset_hidden_state()
                    loop_batch_size = len(inputs)

                    valid = Variable(FloatTensor(loop_batch_size, 1).fill_(1.0), requires_grad=False).cuda()
                    fake = Variable(FloatTensor(loop_batch_size, 1).fill_(0.0), requires_grad=False).cuda()

                    noise = Variable(FloatTensor(np.random.normal(0, 1, (loop_batch_size, noise_dim)))).cuda()
                    semantic_true = att[labels].cuda()
 
                    optimizer_D.zero_grad()

############## New Dataset training #######################
                    true_features_2048 = model(image_sequences)
                    validity_real = discriminator(true_features_2048.detach()).view(-1)
                    d_real_loss = adversarial_loss(validity_real, valid)
                    d_real_loss.backward(retain_graph = True)

############## All Fake Batch Training #######################
                    gen_imgs = generator(semantic_true.float(), noise)
                    validity_fake = discriminator(gen_imgs.detach()).view(-1)
                    d_fake_loss = adversarial_loss(validity_fake, fake)
                    d_fake_loss.backward(retain_graph = True)            
                    optimizer_D.step()

############## Generator training ########################
                    optimizer_G.zero_grad()
                    validity = discriminator(gen_imgs).view(-1)
                    g_loss = adversarial_loss(validity, valid)
                    g_loss.backward(retain_graph = True)
                    optimizer_G.step()                

############## Old Dataset Training ###########################

                    old_labels = Variable(LongTensor(np.random.randint(0, num_classes, loop_batch_size))).cuda()
                    noise = Variable(FloatTensor(np.random.normal(0, 1, (loop_batch_size, noise_dim)))).cuda()
                    semantic_true = att[old_labels].cuda()
 
                    optimizer_D.zero_grad()

                    true_features_2048 = generator1(semantic_true.float(), noise)
                    validity_real = discriminator(true_features_2048.detach()).view(-1)
                    d_real_loss = adversarial_loss(validity_real, valid)
                    d_real_loss.backward(retain_graph = True)

############## All Fake Batch Training #######################
                    gen_imgs = generator(semantic_true.float(), noise)
                    validity_fake = discriminator(gen_imgs.detach()).view(-1)
                    d_fake_loss = adversarial_loss(validity_fake, fake)
                    d_fake_loss.backward(retain_graph = True)            
                    optimizer_D.step()

############## Generator training ########################
                    optimizer_G.zero_grad()
                    validity = discriminator(gen_imgs).view(-1)
                    g_loss = adversarial_loss(validity, valid) + 25*nn.MSELoss()(gen_imgs, true_features_2048)
                    g_loss.backward(retain_graph = True)
                    optimizer_G.step()   

                    _, _, dataset_labels = next(iter(old_train_dataloader))
                    loop_batch_size = len(dataset_labels)
                    dataset_labels = dataset_labels.cuda()
                    noise = Variable(FloatTensor(np.random.normal(0, 1, (loop_batch_size, noise_dim)))).cuda()
                    semantic_true = att[dataset_labels].cuda()
 
                    valid = Variable(FloatTensor(loop_batch_size, 1).fill_(1.0), requires_grad=False).cuda()
                    fake = Variable(FloatTensor(loop_batch_size, 1).fill_(0.0), requires_grad=False).cuda()

                    optimizer_D.zero_grad()

                    dataset_true_features_2048 = generator1(semantic_true.float(), noise)
                    validity_real = discriminator(dataset_true_features_2048.detach()).view(-1)
                    d_real_loss = adversarial_loss(validity_real, valid)
                    d_real_loss.backward(retain_graph = True)

############## All Fake Batch Training #######################
                    dataset_imgs = generator(semantic_true.float(), noise)
                    validity_fake = discriminator(dataset_imgs.detach()).view(-1)
                    d_fake_loss = adversarial_loss(validity_fake, fake)
                    d_fake_loss.backward(retain_graph = True)            
                    optimizer_D.step()

############## Generator training ########################
                    optimizer_G.zero_grad()
                    validity = discriminator(dataset_imgs).view(-1)
                    g_loss = adversarial_loss(validity, valid) + 25*nn.MSELoss()(dataset_true_features_2048, dataset_imgs)
                    g_loss.backward(retain_graph = True)
                    optimizer_G.step()   

                    dataset_logits = classifier(dataset_imgs)
                    old_logits = classifier(gen_imgs)   
                    new_logits = classifier(true_features_2048)

                    #if (args.distillation):
                        #model1.lstm.reset_hidden_state()
                        #dist_target = model1(image_sequences)
                        #dist_target1 = classifier1(dist_target)
                        #logits_dist = predictions[:, :num_classes]
                        #dist_loss = MultiClassCrossEntropy(logits_dist, dist_target1, 0.5)
                        #loss = args.importance*dist_loss + loss
                        #optimizer.zero_grad()
                        #loss.backward(retain_graph = True)
                        #optimizer.step()

                        #gen_imgs1 = generator1(semantic.float(), noise)
                        #gen_imgs1 = generator1(semantic_true.float(), noise)
                        #gen_dist_loss = MultiClassCrossEntropy(gen_imgs, gen_imgs1, 0.5)
                        #gen_dist_loss = nn.MSELoss()(gen_imgs, gen_imgs1)
                        #loss = args.importance*gen_dist_loss
                        #optimizer_G1.zero_grad()
                        #loss.backward(retain_graph = True)
                        #optimizer_G1.step()

                        #validity_fake1 = discriminator1(gen_imgs1.detach()).view(-1)
                        #validity_real1 = discriminator1(dist_target).view(-1)
                        #fake_dist_loss = nn.MSELoss()(validity_fake1, validity_fake)
                        #real_dist_loss = nn.MSELoss()(validity_real1, validity_real)
                        #loss = args.importance*(fake_dist_loss + real_dist_loss)
                        #optimizer_D1.zero_grad()
                        #loss.backward()
                        #optimizer_D1.step()

                    #else:
                        #loss = cls_loss
                        #optimizer.zero_grad()
                        #loss.backward()
                        #optimizer.step()


                    _, old_predictions = torch.max(torch.softmax(old_logits, dim = 1), dim = 1, keepdim = False)
                    _, new_predictions = torch.max(torch.softmax(new_logits, dim = 1), dim = 1, keepdim = False)         
                    _, dataset_predictions = torch.max(torch.softmax(dataset_logits, dim = 1), dim = 1, keepdim = False)         
                    running_old_corrects += torch.sum(old_predictions == old_labels.data)
                    running_new_corrects += torch.sum(new_predictions == labels.data)
                    running_old_corrects += torch.sum(dataset_predictions == dataset_labels.data) 

                old_epoch_acc = running_old_corrects.double() / (2*trainval_sizes['train'])
                new_epoch_acc = running_new_corrects.double() / trainval_sizes['train']

                words = 'data/gen_old_train_acc_epoch' + str(i)
                writer.add_scalar(words, old_epoch_acc, epoch)
                words = 'data/gen_new_train_acc_epoch' + str(i)
                writer.add_scalar(words, new_epoch_acc, epoch)

                print("Set: {} Epoch: {}/{} Train GAN Old Acc: {} Train GAN New Acc: {}".format(i, epoch+1, num_epochs, old_epoch_acc, new_epoch_acc))




                if useTest and epoch % test_interval == (test_interval - 1):
                    classifier.eval()
                    generator.eval()
                    
                    running_old_corrects = 0.0
                    running_new_corrects = 0.0
    
                    for indices, inputs, labels in test_dataloader:
                        labels = Variable(labels.to(device), requires_grad=False)                
                        loop_batch_size = len(inputs)
                        noise = Variable(FloatTensor(np.random.normal(0, 1, (loop_batch_size, noise_dim)))).cuda()
                   
                        semantic = att[labels].cuda()
                        new_features = generator(semantic.float(), noise)
                        new_logits = classifier(new_features)
     
                        _, new_predictions = torch.max(torch.softmax(new_logits, dim = 1), dim = 1, keepdim = False)         
                        running_new_corrects += torch.sum(new_predictions == labels.data)

                    new_epoch_acc = running_new_corrects.double() / len(test_dataloader.dataset)

                    words = 'data/gen_new_test_acc_epoch' + str(i)
                    writer.add_scalar(words, new_epoch_acc, epoch)


                    for indices, inputs, labels in old_test_dataloader:
                        labels = Variable(labels.to(device), requires_grad=False)                
                        loop_batch_size = len(inputs)
                        noise = Variable(FloatTensor(np.random.normal(0, 1, (loop_batch_size, noise_dim)))).cuda()
                   
                        semantic = att[labels].cuda()
                        old_features = generator(semantic.float(), noise)
                        old_logits = classifier(old_features)
     
                        _, old_predictions = torch.max(torch.softmax(old_logits, dim = 1), dim = 1, keepdim = False)         
                        running_old_corrects += torch.sum(old_predictions == labels.data)

                    old_epoch_acc = running_old_corrects.double() / len(old_test_dataloader.dataset)

                    words = 'data/gen_old_test_acc_epoch' + str(i)
                    writer.add_scalar(words, old_epoch_acc, epoch)

                print("Set: {} Epoch: {}/{} GAN Test Old Acc: {} GAN Test New Acc: {}".format(i, epoch+1, num_epochs, old_epoch_acc, new_epoch_acc))



                if (epoch % save_epoch == save_epoch - 1):
                    save_path = os.path.join(save_dir, saveName + '_increment_' + str(i) + '_epoch-' + str(epoch) + '.pth.tar')
                    torch.save({
                        'extractor_state_dict': model.state_dict(),
                        'classifier_state_dict': classifier.state_dict(),
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
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
        
    writer.close()

if __name__ == "__main__":
    train_model()
    print("total_time_taken:",int(-(std_start_time - time.time())/3600)," hrs  ", int(-(std_start_time - time.time())/60%60), " mins")    