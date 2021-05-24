
# -*- coding: utf-8 -*-
"""Tsne.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/11iqaTgnE4Hcxo0BYwALq2io9TFaSTxfO
"""

import os

import torch
import pdb
import numpy as np
from scipy import io
# !pip install MulticoreTSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import time
from models.nets import *
from model import load_pretrained_model, generate_model
from video_data_loader import video_dataset, old_video_dataset

num_classes = 10

feat_path = "convlstm_feat_labs_" + str(num_classes) + ".npy"
convlstm_feat = np.load(feat_path)
#convlstm_feat = convlstm_feat.squeeze_(0)

print("Conv LSTM feature shape {}".format(convlstm_feat.shape))

all_features = convlstm_feat
dataset_label = np.zeros((all_features.shape[0],1))

for i in range(all_features.shape[0]):
    dataset_label[i,:] = all_features[i, -1]
    print(all_features[i,-1])

start_time = time.time()

tsne = TSNE(n_jobs=16)

embeddings = tsne.fit_transform(all_features)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", num_classes)

plot = sns.scatterplot(vis_x, vis_y, hue=dataset_label[:,0], legend='full', palette=palette)
#plt.savefig("2048_tsne.png")
plt.savefig("dataset_tsne.png")
print("--- {} mins {} secs---".format((time.time() - start_time)//60,(time.time() - start_time)%60))