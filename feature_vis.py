import os
import pdb
import numpy as np
from scipy import io
# !pip install MulticoreTSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import time

num_classes = 10
start_class = 40

feat_path = f"gen_features/episode_0/convlstm_feat_labs_{num_classes}_{start_class}.npy"
convlstm_feat = np.load(feat_path)

gen_feat_path = f"gen_features/episode_0/gen_feat_labs_{num_classes}_{start_class}.npy"
gen_feat = np.load(gen_feat_path)

labels = np.concatenate([convlstm_feat[:,-1], gen_feat[:,-1]])

print("Conv LSTM feature shape {}".format(convlstm_feat.shape))
print("Generator feature shape {}".format(gen_feat.shape))

convlstm_feat = PCA(n_components = gen_feat.shape[1]).fit_transform(convlstm_feat)
all_features = np.concatenate((convlstm_feat, gen_feat), 0)
dataset_label = np.zeros((all_features.shape[0],1))
dataset_style = np.zeros((all_features.shape[0],1))

dataset_style[convlstm_feat.shape[0]:,:] = 1

dataset_label[:,0] = labels
#dataset_label[convlstm_feat.shape[0]:,:] = 1

start_time = time.time()

tsne = TSNE(n_jobs=16)

embeddings = tsne.fit_transform(all_features)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", num_classes)
# Legend symbols
# https://matplotlib.org/stable/api/markers_api.html

plot = sns.scatterplot(vis_x, vis_y, hue=dataset_label[:,0], style = dataset_style[:,0], markers=['P', 'o'], palette=palette)
plot.get_legend().set_title("Classes")
handles, labels = plot.get_legend_handles_labels()
#labels[-1] = "gen"
#labels[-2] = "conv"
plot.legend(handles, labels) 
plt.savefig("dataset_tsne.png")
print("--- {} mins {} secs---".format((time.time() - start_time)//60,(time.time() - start_time)%60))
