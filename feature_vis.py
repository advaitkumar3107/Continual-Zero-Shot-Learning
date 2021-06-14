import os
import torch
import pdb
import numpy as np
from scipy import io
# !pip install MulticoreTSNE
#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import time
from models.nets import *
from model import load_pretrained_model, generate_model
from video_data_loader import video_dataset, old_video_dataset

num_classes = 10

feat_path = "gen_features/episode_0/convlstm_feat_labs_" + str(num_classes) + "_0.npy"
convlstm_feat = np.load(feat_path)

gen_feat_path = "gen_features/episode_0/gen_feat_labs_" + str(num_classes) + "_0.npy"
gen_feat = np.load(gen_feat_path)

print("Conv LSTM feature shape {}".format(convlstm_feat.shape))
print("Generator feature shape {}".format(gen_feat.shape))

x = np.concatenate((convlstm_feat, gen_feat), 0)
y = x[:,-1]
x = x[:,:-1]

tsne = TSNE(n_components=2, verbose=1, perplexity=25, n_iter=1000, learning_rate=200)    
embeddings = tsne.fit_transform(x)
sns.set(rc={'figure.figsize':(11.7,8.27)})

vis_x = embeddings[:,0]
vis_y = embeddings[:,1]
palette = sns.color_palette("bright", num_classes)

plot = sns.scatterplot(vis_x, vis_y, hue = y, legend = 'full', palette = palette)

plt.savefig("dataset_tsne.png")