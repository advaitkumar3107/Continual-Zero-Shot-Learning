from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from coclust.evaluation.external import accuracy

import matplotlib.pyplot as plt
import numpy as np

def k_means_clusters(feat_path, num_clusters):
    gen_feat = np.load(feat_path)
    data = gen_feat[:,:-1].astype(np.double)

    print("Generator feature shape {}".format(gen_feat.shape))

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init="k-means++", n_clusters=num_clusters)
    kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation="nearest",
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired, aspect="auto", origin="lower")

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
            color="w", zorder=10)
    plt.title("K-means clustering on the unseen data (PCA-reduced data)\n"
          "Centroids are marked with white cross")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("k_means.png")

    return kmeans.labels_

def clustering_accuracy(labels, predicted_labels, offset = 40):
    labels = labels - 40
    acc = accuracy(labels, predicted_labels)
    return acc

if __name__ == '__main__':
    num_classes = 10
    start_class = 40
    gen_feat_path = f"gen_features/episode_0/gen_feat_labs_{num_classes}_{start_class}.npy"
    #gen_feat_path = f"gen_features/episode_0/convlstm_feat_labs_{num_classes}_{start_class}.npy"
    gen_feat = np.load(gen_feat_path)

    labels = gen_feat[:,-1].astype(np.int)
    z = k_means_clusters(gen_feat_path, 10)
    print(clustering_accuracy(labels, z))
