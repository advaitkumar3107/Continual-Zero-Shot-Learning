import os

import torch
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


total_classes = 40
all_classes = np.arange(total_classes)
train_dataset = video_dataset(train = True, classes = all_classes[:total_classes])
train_dataloader = DataLoader(train_dataset, batch_size = 16, shuffle = False, num_workers = 0)

model = generate_model()
model = load_pretrained_model(model, 'saved_weights/resnet_50.pth')
for param in model.parameters():
    param.requires_grad = False
model = nn.Sequential(*list(model.children())[:-1])
model = model.cuda()
model.eval()


for i, (_, inputs, labels) in enumerate(train_dataloader):
	print(i)
	# torch.cuda.empty_cache()
	rand_int = np.random.randint(48)
	inputs = inputs.permute(0,1,2,3,4).cuda()
	print(inputs.shape)
	batch_size = inputs.size(0)
	convlstm_feats = model(inputs)
	convlstm_feats = convlstm_feats.contiguous().view(convlstm_feats.size(0), -1)
	if (i == 0):		
		convlstm_feat_labs = torch.cat((convlstm_feats,labels.type(torch.cuda.FloatTensor).unsqueeze(1)), dim =1)
	else:
		convlstm_feat_labs = torch.cat((convlstm_feat_labs, torch.cat((convlstm_feats,labels.type(torch.cuda.FloatTensor).unsqueeze(1)), dim =1)), 0)
	# pdb.set_trace()	
	print(convlstm_feat_labs.shape)

np.save(f"convlstm_feat_labs_{total_classes}.npy", convlstm_feat_labs.cpu().detach().numpy())
