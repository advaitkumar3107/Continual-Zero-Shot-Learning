import os

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

total_classes = 40
all_classes = range(total_classes)

train_dataloader, test_dataloader, _, _ = create_data_loader('ucf101_i3d/i3d.mat', all_classes[:40])

model = Modified_Generator(300, 1024)
checkpoint = torch.load(os.path.join('run/pipeline_set1/' + 'Bi-LSTM-ucf101_increment_epoch-299' + '.pth.tar'),
                       map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['generator_state_dict'])
model = model.cuda()
model.eval()

att = np.load("../npy_files/seen_semantic_51.npy")
att = torch.tensor(att).cuda()  

print(att.shape)

for i, (inputs, labels) in enumerate(train_dataloader):
    print(i)
	# torch.cuda.empty_cache()
    batch_size = inputs.size(0)
    labels = Variable(labels, requires_grad = False).long().cuda()
    noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, 1024)))).cuda()
    semantic = att[labels]
    convlstm_feats = model(semantic.float(), noise)
    convlstm_feats = convlstm_feats.contiguous().view(convlstm_feats.size(0), -1)
    if (i == 0):		
        convlstm_feat_labs = torch.cat((convlstm_feats,labels.type(torch.cuda.FloatTensor).unsqueeze(1)), dim =1)
    else:
        convlstm_feat_labs = torch.cat((convlstm_feat_labs, torch.cat((convlstm_feats,labels.type(torch.cuda.FloatTensor).unsqueeze(1)), dim =1)), 0)
	# pdb.set_trace()	
    print(convlstm_feat_labs.shape)

np.save(f"convlstm_feat_labs_{total_classes}_test.npy", convlstm_feat_labs.cpu().detach().numpy())
