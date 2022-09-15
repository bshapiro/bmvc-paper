import autograd.numpy as np
import autograd.scipy.stats.multivariate_normal as mvn
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import seaborn as sns
from copy import deepcopy


def all_same(items):
    return all(np.array_equal(x, items[0]) for x in items)


def size(item):
    try:
        return max(item.shape[0], item.shape[1])
    except:
        try:
            return len(item)
        except:
            return 1


def softplus(value):
    if size(value) > 1:
        # import pdb; pdb.set_trace()
        return np.array([softplus(item) for item in value])
    else:
        return threshold(value)


def hardplus(value):
    if size(value) > 1:
        return np.array([hardplus(item) for item in value])
    else:
        if value <= 0:
            return 0
        else:
            return value


def regularize(item, diagonal=False):
    if diagonal:
        return item + np.identity(item.shape[0]) * 0.000001
    else:
        return item + 0.000001


def threshold(item):
    if item < 600:
        try:
            return np.log(np.exp(item) + 1)
        except RuntimeWarning:
            import pdb; pdb.set_trace()
    else:
        return item


def visualize_R(R):
    ax = sns.heatmap(R, cmap='Blues', cbar=True, xticklabels=True, yticklabels=True, vmin=0)
    plt.title('R')
    plt.show()


def visualize_2d_results(samples1, samples2, labels1, labels2, means1, means2, covs1, covs2):
    plot_clusters(None, samples1, labels1, 'View 1')
    ax = plt.gca()
    for ci in range(len(means1)):
        variational_contour = lambda x: mvn.pdf(x, means1[ci], covs1[ci])
        plot_isocontours(ax, variational_contour)
    plt.grid()
    plt.legend()
    plt.show()

    plot_clusters(None, samples2, labels2, 'View 2')
    ax = plt.gca()
    for ci in range(len(means2)):
        variational_contour = lambda x: mvn.pdf(x, means2[ci], covs2[ci])
        plot_isocontours(ax, variational_contour)

    plt.grid()
    plt.legend()
    plt.show()


def plot_clusters(clusters, samples, labels, title):
    sample_x = [sample[0] for sample in samples]
    sample_y = [sample[1] for sample in samples]
    labels = labels
    df = pd.DataFrame({"x": sample_x, "y": sample_y, "cluster": labels})
    groups = df.groupby("cluster")
    plt.title(title)
    for name, group in groups:
        plt.scatter(group["x"], group["y"], label=name)
    if clusters is not None:
        plt.scatter([cluster[0] for cluster in clusters], [cluster[1] for cluster in clusters], s=50, c='black')


def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-5, 5], numticks=101):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, levels=[0.05, 0.25, 0.5, 0.75])
    ax.set_yticks([])
    ax.set_xticks([])


def compute_affinity(Z1, Z2, R, K, L):
    nonzero_RA = defaultdict(list)
    nonzero_RB = defaultdict(list)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] == 1:
                nonzero_RA[i].append(j)
                nonzero_RB[j].append(i)

    total_related_1 = 0
    total_related_2 = 0
    matrix = np.zeros((K, L), dtype=float)
    for cluster_label1 in range(K):
        indices1 = set([i for i in range(len(Z1)) if Z1[i] == cluster_label1])
        for cluster_label2 in range(L):
            # print(cluster_label1, cluster_label2)
            indices2 = set([j for j in range(len(Z2)) if Z2[j] == cluster_label2])
            if len(indices1) == 0 or len(indices2) == 0:
                continue
            related_indices1 = []
            related_indices2 = []
            for i in indices1:
                related_indices1.extend(nonzero_RA[i])
            for j in indices2:
                related_indices2.extend(nonzero_RB[j])
            cluster_relations_1 = len(set(related_indices1).intersection(indices2))
            cluster_relations_2 = len(set(related_indices2).intersection(indices1))
            cluster_relations = cluster_relations_1 + cluster_relations_2

            if cluster_label1 == cluster_label2:
                total_related_1 += cluster_relations_1
                total_related_2 += cluster_relations_2

            all_relations = len(related_indices1) + len(related_indices2)
            if all_relations == 0:
                matrix[cluster_label1, cluster_label2] = 0
            else:
                matrix[cluster_label1, cluster_label2] = cluster_relations / all_relations
                matrix[cluster_label1, cluster_label2] = cluster_relations
                matrix[cluster_label1, cluster_label2] = ((cluster_relations_1 / (len(related_indices1) + 0.0001)) + (cluster_relations_2 / (len(related_indices2) + 0.0001))) / 2.0
    # print("Total view 1 relationships fused:", total_related_1)
    # print("Total view 2 relationships fused:", total_related_2)
    return matrix
