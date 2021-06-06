import os
import argparse
import torch
from torch import FloatTensor
import pdb
import numpy as np
from scipy import io
# !pip install MulticoreTSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import time
from models.nets import *
from model import load_pretrained_model, generate_model
from video_data_loader import video_dataset, old_video_dataset
from dataloader import create_data_loader

parser.add_argument('--load_name', type = str, default = "temp", help = 'Name of the directory which contains the saved weights')
parser.add_argument('--gpu', type=int, default=3,
                    help='GPU ID, start from 0')
parser.add_argument('--increment_class', type = int, default = 10, help = 'Number of classes to increment by')
parser.add_argument('--num_class', type = int, default = 20, help = 'Number of classes in classifier')
parser.add_argument('--resume_epoch', type = int, default = None, help = 'Epoch from where to load weights')
parser.add_argument('--feat_path', type = str, default = "ucf101_i3d/i3d.mat", help = 'Path which contains the pretrained feats')
parser.add_argument('--att_path', type = str, default = "ucf101_i3d/split_1/att_splits.mat", help = 'Path which contains the pretrained attributes')
parser.add_argument('--increment', type = int, default = None, help = 'Number of increments the model was trained for')
parser.add_argument('--dataset', type = str, default = "ucf101", help = 'Dataset to test on')
parser.add_argument('--save_name', type = str, default = "episode_0", help = 'Name of the directory to save features')

args = parser.parse_args()

total_classes = args.num_class
all_classes = range(total_classes)
increments = total_classes // args.increment_class

att_path = args.att_path
feat_path = args.feat_path

model = Modified_Generator(300, 8192)

if (args.increment is None):
    checkpoint = torch.load(os.path.join('run/' + load_name + '/Bi-LSTM-' + args.dataset + '_increment_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)

else:
    checkpoint = torch.load(os.path.join('run/' + load_name + '/Bi-LSTM-' + args.dataset + '_increment_' + str(args.increment) + '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)

model.load_state_dict(checkpoint['generator_state_dict'])
model = model.cuda()

feats = sio.loadmat(att_path)
att = feats['att']
att = np.transpose(att, (1,0))
att = torch.tensor(att).cuda()  

classes = 0

for i in range(increments):
    train_dataloader, _, _, _ = create_data_loader(feat_path, all_classes[classes:classes+args.increment_class])

    for i, (inputs, labels) in enumerate(train_dataloader):
        batch_size = inputs.size(0)
        labels = Variable(labels, requires_grad = False).long().cuda()
        noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, 1024)))).cuda()
        semantic = att[labels]
        convlstm_feats = inputs.cuda()
        convlstm_feats = convlstm_feats.contiguous().view(convlstm_feats.size(0), -1)
        gen_feats = model(semantic.float(), noise)
        gen_feats = gen_feats.contiguous().view(gen_feats.size(0), -1)
        if (i == 0):		
            convlstm_feat_labs = torch.cat((convlstm_feats.float(), labels.float().unsqueeze(1)), dim = 1)
            gen_feat_labs = torch.cat((gen_feats, labels.type(torch.cuda.FloatTensor).unsqueeze(1)), dim =1)
        else:
            convlstm_feat_labs = torch.cat((convlstm_feat_labs.float(), torch.cat((convlstm_feats.float(),labels.float().unsqueeze(1)), dim=1)), 0)
            gen_feat_labs = torch.cat((gen_feat_labs, torch.cat((gen_feats,labels.float().unsqueeze(1)), dim =1)), 0)
    	# pdb.set_trace()	
        print(convlstm_feat_labs.shape)
        print(gen_feat_labs.shape)

    np.save(f"{args.save_name}/convlstm_feat_labs_{args.increment_class}.npy", convlstm_feat_labs.cpu().detach().numpy())
    np.save(f"{args.save_name}/gen_feat_labs_{args.increment_class}.npy", gen_feat_labs.cpu().detach().numpy())
    classes += args.increment_class