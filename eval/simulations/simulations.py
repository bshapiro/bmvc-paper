from numpy.random import multivariate_normal, binomial, normal
import autograd.scipy.stats.multivariate_normal as mvn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score
import pandas as pd
from copy import copy
from random import seed, sample


def two_cluster_sim(n, sigma):
    clusters = [[-2, 0], [2, 0]]
    ns = [n, n]
    R = np.eye(np.sum(ns))
    sigmas = [sigma, sigma]
    return clusters, ns, ns, sigmas, sigmas, R


def two_cluster_sim_onto(n, sigma):
    clusters = [[-2, 0], [2, 0]]
    ns1 = [2 * n, 2 * n]
    ns2 = [int(n), int(n)]

    R1 = np.eye(ns2[0])
    R2 = copy(R1)
    R11 = np.concatenate((R1, R2))
    R12 = np.zeros(R11.shape)
    R21 = np.zeros(R11.shape)
    R22 = copy(R11)
    R = np.concatenate((np.concatenate((R11, R12)).T, np.concatenate((R21, R22)).T)).T

    sigmas = [sigma, sigma]

    return clusters, ns1, ns2, sigmas, sigmas, R


def two_cluster_sim_missing(n, sigma):
    clusters, ns1, ns2, sigmas1, sigmas2, R = two_cluster_sim(n, sigma)
    missing = sample(range(n * 2), n)
    for i in missing:
        R[i, i] = 0
    return clusters, ns1, ns2, sigmas1, sigmas2, R


def four_cluster_sim(n, sigma):
    clusters = [[-2, 0], [2, 0], [-2, -4], [2, -4]]
    ns1 = [n, n, n, n]
    ns2 = [n, n, n, n]
    R = np.eye(n * 4)

    sigmas1 = [sigma, sigma, sigma, sigma]
    sigmas2 = [sigma, sigma, sigma, sigma]
    return clusters, ns1, ns2, sigmas1, sigmas2, R


def four_cluster_baseline(n, sigma):
    clusters = [[-2, 0], [2, 0], [-2, -4], [2, -4]]
    ns1 = [n, n, n, n]
    ns2 = [n, n, n, n]
    R = np.eye(n * 4)

    sigmas1 = [sigma, sigma, sigma, sigma]
    sigmas2 = [sigma, sigma, sigma, sigma]
    return clusters, ns1, ns2, sigmas1, sigmas2, R


def generate_data_baseline(clusters, ns, sigmas, p, rs=None):
    random_state = np.random.RandomState(rs)

    samples = [[], []]
    labels = [[], []]

    # generate 1st view
    for group in [0, 1, 2, 3]:
        samples[0].extend(random_state.multivariate_normal(clusters[group], sigmas[0][group] * np.eye(len(clusters[group])), size=ns[0][group]))
        labels[0].extend(ns[0][group] * [group])

    # generate 2nd view
    ps = [0, 0, p, 1 - p]
    for group in [0, 1]:
        samples[1].extend(random_state.multivariate_normal(clusters[group], sigmas[1][group] * np.eye(len(clusters[group])), size=ns[1][group]))
        labels[1].extend(ns[0][group] * [group])
    for group in [2, 3]:
        n_cluster = ns[1][group]
        bin_sample = random_state.binomial(1, ps[group], n_cluster) + 2
        for j in bin_sample:
            samples[1].append(random_state.multivariate_normal(clusters[j], sigmas[1][j] * np.eye(len(clusters[j]))))
        labels[1].extend(bin_sample)

    return samples[0], samples[1], labels[0], labels[1]


def generate_data_baseline_mod(clusters, ns, sigmas, p, mod_size, num_mod, rs=None):
    random_state = np.random.RandomState(rs)

    samples = [[], []]
    labels = [[], []]

    # generate 1st view
    for group in [0, 1, 2, 3]:
        samples[0].extend(random_state.multivariate_normal(clusters[group], sigmas[0][group] * np.eye(len(clusters[group])), size=ns[0][group]))
        labels[0].extend(ns[0][group] * [group])

    # generate 2nd view
    ps = [0, 0, p, 1 - p]
    for group in [0, 1]:
        samples[1].extend(random_state.multivariate_normal(clusters[group], sigmas[1][group] * np.eye(len(clusters[group])), size=ns[1][group]))
        labels[1].extend(ns[0][group] * [group])

    seed(rs)
    random_sample2 = sample(list(np.argwhere(np.array(labels[1]) == 0).astype(int).flatten()), num_mod)
    distance = clusters[1][0] - clusters[0][0]
    shift = random_state.normal(mod_size, 0.05)
    for index in random_sample2:
        samples[1][index][0] = samples[1][index][0] + shift * distance

    for group in [2, 3]:
        n_cluster = ns[1][group]
        bin_sample = random_state.binomial(1, ps[group], n_cluster) + 2
        for j in bin_sample:
            samples[1].append(random_state.multivariate_normal(clusters[j], sigmas[1][j] * np.eye(len(clusters[j]))))
        labels[1].extend(bin_sample)

    return samples[0], samples[1], labels[0], labels[1]


def generate_data_p(clusters, ns, sigmas, p):
    samples = [[], []]
    labels = [[], []]
    ps = [p, 1 - p]
    for view in [0, 1]:
        for group in [0, 1]:
            n_cluster = ns[view][group]
            bin_sample = binomial(1, ps[group], n_cluster)
            for j in bin_sample:
                samples[view].append(multivariate_normal(clusters[j], sigmas[view][j] * np.eye(len(clusters[j]))))
            labels[view].extend(bin_sample)

    return samples[0], samples[1], labels[0], labels[1]


def generate_data_p_gt(clusters, ns, sigmas, p, rs=None):
    random_state = np.random.RandomState(rs)

    samples = [[], []]
    labels = [[], []]

    # generate 1st view
    for group in [0, 1]:
        samples[0].extend(random_state.multivariate_normal(clusters[group], sigmas[0][group] * np.eye(len(clusters[group])), size=ns[0][group]))
        labels[0].extend(ns[0][group] * [group])

    # generate 2nd view
    ps = [p, 1 - p]
    for group in [0, 1]:
        n_cluster = ns[1][group]
        bin_sample = random_state.binomial(1, ps[group], n_cluster)
        for j in bin_sample:
            samples[1].append(random_state.multivariate_normal(clusters[j], sigmas[1][j] * np.eye(len(clusters[j]))))
        labels[1].extend(bin_sample)

    return samples[0], samples[1], labels[0], labels[1]


def generate_data_p_gt_3(clusters, ns, sigmas, p):
    samples = [[], []]
    labels = [[], []]

    # generate 1st view
    for group in [0, 1]:
        samples[0].extend(multivariate_normal(clusters[group], sigmas[0][group] * np.eye(len(clusters[group])), size=ns[0][group]))
        labels[0].extend(ns[0][group] * [group])

    # generate 2nd view
    ps = [p, 1 - p]

    group = 0
    n_cluster = ns[1][group]
    if group < 2:
        bin_sample = binomial(1, ps[group], n_cluster)
        for j in bin_sample:
            samples[1].append(multivariate_normal(clusters[j], sigmas[1][j] * np.eye(len(clusters[j]))))
        labels[1].extend(bin_sample)

    samples[1].extend(multivariate_normal(clusters[2], sigmas[1][2] * np.eye(len(clusters[2])), size=int(ns[1][2] / 2)))
    labels[1].extend(int(ns[1][2] / 2) * [2])

    group = 1
    n_cluster = ns[1][group]
    if group < 2:
        bin_sample = binomial(1, ps[group], n_cluster)
        for j in bin_sample:
            samples[1].append(multivariate_normal(clusters[j], sigmas[1][j] * np.eye(len(clusters[j]))))
        labels[1].extend(bin_sample)

    samples[1].extend(multivariate_normal(clusters[2], sigmas[1][2] * np.eye(len(clusters[2])), size=int(ns[1][2] / 2)))
    labels[1].extend(int(ns[1][2] / 2) * [2])

    return samples[0], samples[1], labels[0], labels[1]


def generate_data_p_gt_4(clusters, ns, sigmas, p):
    samples = [[], []]
    labels = [[], []]

    # generate 1st view
    for group in [0, 1]:
        samples[0].extend(multivariate_normal(clusters[group], sigmas[0][group] * np.eye(len(clusters[group])), size=ns[0][group]))
        labels[0].extend(ns[0][group] * [group])
    samples[0].extend(multivariate_normal(clusters[2], sigmas[0][2] * np.eye(len(clusters[2])), size=ns[0][2]))
    samples[0].extend(multivariate_normal(clusters[3], sigmas[0][3] * np.eye(len(clusters[3])), size=ns[0][3]))
    labels[0].extend(ns[0][2] * [2])
    labels[0].extend(ns[0][3] * [3])
    # generate 2nd view
    ps = [p, 1 - p]
    for group in [0, 1]:
        n_cluster = ns[1][group]
        bin_sample = binomial(1, ps[group], n_cluster)
        for j in bin_sample:
            samples[1].append(multivariate_normal(clusters[j], sigmas[1][j] * np.eye(len(clusters[j]))))
        labels[1].extend(bin_sample)

    samples[1].extend(multivariate_normal(clusters[2], sigmas[1][2] * np.eye(len(clusters[2])), size=int(ns[1][2] * 0.5)))
    samples[1].extend(multivariate_normal(clusters[3], sigmas[1][3] * np.eye(len(clusters[3])), size=int(ns[1][3] * 0.5)))
    samples[1].extend(multivariate_normal(clusters[2], sigmas[1][2] * np.eye(len(clusters[2])), size=int(ns[1][2] * 0.5)))
    samples[1].extend(multivariate_normal(clusters[3], sigmas[1][3] * np.eye(len(clusters[3])), size=int(ns[1][3] * 0.5)))
    labels[1].extend(int(ns[1][2] * 0.5) * [2])
    labels[1].extend(int(ns[1][3] * 0.5) * [3])
    labels[1].extend(int(ns[1][2] * 0.5) * [2])
    labels[1].extend(int(ns[1][3] * 0.5) * [3])

    return samples[0], samples[1], labels[0], labels[1]


def generate_data_p_mod(clusters, ns, sigmas, p, mod_size, num_mod):
    samples1, samples2, labels1, labels2 = generate_data_p(clusters, ns, sigmas, p)
    random_sample1 = sample(list(np.argwhere(np.logical_not(labels1).astype(int)).flatten()), num_mod)
    random_sample2 = sample(list(np.argwhere(np.logical_not(labels2).astype(int)).flatten()), num_mod)
    distance = clusters[1][0] - clusters[0][0]
    shift = normal(mod_size, 0.05)
    for index in random_sample1:
        samples1[index][0] = samples1[index][0] + shift * distance
    for index in random_sample2:
        samples2[index][0] = samples2[index][0] + shift * distance
    return samples1, samples2, labels1, labels2


def generate_data_p_mod_gt(clusters, ns, sigmas, p, mod_size, num_mod, rs=None):
    random_state = np.random.RandomState(rs)
    seed(rs)
    samples1, samples2, labels1, labels2 = generate_data_p_gt(clusters, ns, sigmas, p, rs=rs)
    random_sample2 = sample(list(np.argwhere(np.logical_not(labels2).astype(int)).flatten()), num_mod)
    distance = clusters[1][0] - clusters[0][0]
    shift = random_state.normal(mod_size, 0.05)
    for index in random_sample2:
        samples2[index][0] = samples2[index][0] + shift * distance
    return samples1, samples2, labels1, labels2


def generate_data_p_mod_gt_3(clusters, ns, sigmas, p, mod_size, num_mod):
    samples1, samples2, labels1, labels2 = generate_data_p_gt_3(clusters, ns, sigmas, p)
    random_sample2 = sample(list(np.argwhere(np.logical_not(labels2).astype(int)).flatten()), num_mod)
    distance = clusters[1][0] - clusters[0][0]
    shift = normal(mod_size, 0.05)
    for index in random_sample2:
        samples2[index][0] = samples2[index][0] + shift * distance
    return samples1, samples2, labels1, labels2


def generate_data_p_mod_gt_4(clusters, ns, sigmas, p, mod_size, num_mod):
    samples1, samples2, labels1, labels2 = generate_data_p_gt_4(clusters, ns, sigmas, p)
    random_sample2 = sample(list(np.argwhere(np.logical_not(labels2).astype(int)).flatten()), num_mod)
    distance = clusters[1][0] - clusters[0][0]
    shift = normal(mod_size, 0.05)
    for index in random_sample2:
        samples2[index][0] = samples2[index][0] + shift * distance
    return samples1, samples2, labels1, labels2


def plot_clusters(clusters, samples, labels, title):
    label_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    sample_x = [sample[0] for sample in samples]
    sample_y = [sample[1] for sample in samples]
    labels = labels
    df = pd.DataFrame({"x": sample_x, "y": sample_y, "cluster": labels})
    groups = df.groupby("cluster")
    plt.title(title)
    for name, group in groups:
        plt.scatter(group["x"], group["y"], label=label_dict[name])
    if clusters is not None:
        plt.scatter([cluster[0] for cluster in clusters], [cluster[1] for cluster in clusters], s=50, c='black')


def visualize_2d(clusters1, clusters2, samples1, samples2, labels1, labels2, fname=None):
    plt.clf()
    plot_clusters(clusters1, samples1, labels1, '')
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.legend()

    if fname is not None:
        plt.savefig(fname + '_1.png')
    else:
        plt.show()

    plt.clf()
    plot_clusters(clusters2, samples2, labels1, '')
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.legend()

    if fname is not None:
        plt.savefig(fname + '_2.png')
    else:
        plt.show()


def visualize_2d_results(samples1, samples2, labels1, labels2, means1, means2, covs1, covs2, fname=None):
    plt.clf()
    plot_clusters(None, samples1, labels1, 'View 1')
    plt.axis('equal')
    ax = plt.gca()
    for ci in range(len(means1)):
        variational_contour = lambda x: mvn.pdf(x, means1[ci], covs1[ci])
        plot_isocontours(ax, variational_contour)
    plt.grid()
    plt.legend()

    if fname is not None:
        plt.savefig(fname + '_1.png')
    else:
        plt.show()

    plt.clf()
    plot_clusters(None, samples2, labels2, 'View 2')
    plt.axis('equal')
    ax = plt.gca()
    for ci in range(len(means2)):
        variational_contour = lambda x: mvn.pdf(x, means2[ci], covs2[ci])
        plot_isocontours(ax, variational_contour)

    plt.grid()
    plt.legend()

    if fname is not None:
        plt.savefig(fname + '_2.png')
    else:
        plt.show()


def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-2, 2], numticks=101):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z, levels=[0.05, 0.25, 0.5, 0.75])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(b=True)


def compare_clusterings_ari(labels, pred_labels, test_name):
    score = adjusted_rand_score(labels, pred_labels)
    print(test_name, "score:", score)
    return score
