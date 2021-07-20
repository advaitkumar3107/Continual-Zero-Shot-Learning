import os
import pdb
import numpy as np
from scipy import io
# !pip install MulticoreTSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import time

num_classes = 101
start_class = 0

feat_path = f"npy_feats/convlstm_feat_labs_{num_classes}_{start_class}.npy"
convlstm_feat = np.load(feat_path)

gen_feat_path = f"npy_feats/gen_feat_labs_{num_classes}_{start_class}.npy"
gen_feat = np.load(gen_feat_path)

print(convlstm_feat.mean(axis=1))
print(convlstm_feat.mean(axis=0))
print(convlstm_feat.std(axis=0))
print(convlstm_feat.std(axis=1))

pdb.set_trace()

print("Conv LSTM feature shape {}".format(convlstm_feat.shape))
print("Generator feature shape {}".format(gen_feat.shape))

unseen_conv_feats = []
seen_conv_feats = []
for i in range(convlstm_feat.shape[0]):
	if convlstm_feat[i][-1]<=40:
		seen_conv_feats.append(convlstm_feat[i])
	else:
		unseen_conv_feats.append(convlstm_feat[i])
unseen_gen_feats = []
seen_gen_feats = []
for i in range(gen_feat.shape[0]):
	if gen_feat[i][-1]<=40:
		seen_gen_feats.append(gen_feat[i])
	else:
		unseen_gen_feats.append(gen_feat[i])

convlstm_feat = np.array(unseen_conv_feats)
gen_feat = np.array(unseen_gen_feats)
all_features = np.concatenate(( np.array(seen_conv_feats),  np.array(seen_gen_feats)), 0)
# all_features = np.concatenate((convlstm_feat , gen_feat ), 0)
# all_features = np.concatenate((convlstm_feat, gen_feat), 0)
num_classes =  np.unique(np.array(all_features)[:,-1]).shape[0]
dataset_label = np.zeros((all_features.shape[0],1))
dataset_style = np.zeros((all_features.shape[0],1))

dataset_style[convlstm_feat.shape[0]:,:] = 1

for i in range(all_features.shape[0]):
    dataset_label[i,:] = all_features[i, -1]
    # print(all_features[i,-1])

#dataset_label[convlstm_feat.shape[0]:,:] = 1

start_time = time.time()

tsne = TSNE(n_jobs=16)

embeddings = tsne.fit_transform(all_features)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]

sns.set(rc={'figure.figsize':(11.7,8.27)})
# pdb.set_trace()
palette = sns.color_palette("bright", num_classes)

plot = sns.scatterplot(vis_x, vis_y, hue=dataset_label[:,0], style = dataset_style[:,0], markers=['P', 'o'], palette=palette)
plot.get_legend().set_title("Classes")
handles, labels = plot.get_legend_handles_labels()
labels[-1] = "gen"
labels[-2] = "conv"
plot.legend(handles, labels) 
plt.savefig("tsnes/seen_0-40_tsne.png")
print("--- {} mins {} secs---".format((time.time() - start_time)//60,(time.time() - start_time)%60))