from black_box_vi_base import Model
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
from autograd.scipy.stats import gamma
from autograd.numpy.random import RandomState
import autograd.numpy as np
from helpers import softplus, hardplus, regularize, compute_affinity
from collections import Counter
from collections import defaultdict
from scipy.sparse.csgraph import connected_components
from itertools import product
from multiprocessing import Pool
from autograd.scipy import stats
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.stats import invwishart
import seaborn as sns
import matplotlib.pyplot as plt


class BaseMVCModel(Model):

    def __init__(self, view1, view2, R=None, K=5, phi=None, phi_mean_prior=1, phi_scale_prior=0, init='kmeans', random_state=None, inference='alternating', phi_power=1):
        self.view1 = view1
        self.view2 = view2
        self.K = K
        self.D1 = view1.shape[1]
        self.D2 = view2.shape[1]
        self.N = view1.shape[0]
        self.M = view2.shape[0]
        self.R = R
        self.R_dict = dict(zip(range(self.N), [np.nonzero(R[i])[0] for i in range(self.N)]))
        self.phi = phi
        self.phi_scale_prior = float(phi_scale_prior)
        self.phi_mean_prior = float(phi_mean_prior)
        self.inv_wish_v = 2
        self.inv_wish_psi = 1
        self.alpha_A = 0.01
        self.alpha_B = 0.01
        self.converged = False
        self.prev_log_likelihood_M2 = -99999999999
        self.prev_log_likelihood_E = -99999999999
        self.prev_log_likelihood_M1 = -99999999999
        self.prev_log_likelihood_A = -99999999999
        self.prev_phi = None
        self.prev_params = None
        self.num_fused = 0
        self.pseudo = False
        self.phi_power = phi_power
        self.C = np.eye(self.K)

        self.ccs, self.related_dict = self.run_connected_components()

        if init == 'kmeans':
            print("Initializing via kmeans...")
            model1 = KMeans(n_clusters=K, random_state=random_state)
            self.zA = model1.fit_predict(view1)
            self.muA = model1.cluster_centers_
            self.sigmaA = [np.identity(self.D1) for i in range(self.K)]
            model2 = KMeans(n_clusters=K, random_state=random_state)
            self.zB = model2.fit_predict(view2)
            self.muB = model2.cluster_centers_
            self.sigmaB = [np.identity(self.D2) for i in range(self.K)]
            print("Initial clustering:", Counter(self.zA), Counter(self.zB))
            print(self.muA.shape, self.muB.shape)
        elif init == 'gmm':
            print("Initializing via GMM...")
            model1 = GaussianMixture(n_components=K, random_state=random_state, covariance_type='full')
            self.zA = model1.fit_predict(view1)
            self.muA = model1.means_
            self.sigmaA = [np.identity(self.D1) for i in range(self.K)]
            model2 = GaussianMixture(n_components=K, random_state=random_state, covariance_type='full')
            self.zB = model2.fit_predict(view2)
            self.muB = model2.means_
            self.sigmaB = [np.identity(self.D2) for i in range(self.K)]
            print(self.zA, self.zB)
        else:
            print("Initializing randomly...")
            rs = RandomState(seed=random_state)
            cov1 = np.cov(self.view1.T)
            cov2 = np.cov(self.view2.T)
            self.muA = self.view1[rs.choice(range(self.N), size=self.K), :]
            self.muB = self.view2[rs.choice(range(self.M), size=self.K), :]
            self.sigmaA = [cov1 for i in range(self.K)]
            self.sigmaB = [cov2 for i in range(self.K)]
            print(self.muA, self.muB)
            self.reestimate_assignments(np.ones(K) / K, np.ones(K) / K, 1)

        self.zA = np.array(self.zA, dtype=int)
        self.zB = np.array(self.zB, dtype=int)
        self.z_list = [self.zA, self.zB]

        self.inference = inference
        if inference == "parallel":
            self.num_procs = 2
            self.pool = Pool(self.num_procs)
        elif inference == "serial":
            pass
        elif inference == "alternating":
            pass

        if R is None:
            if self.N != self.M:
                raise Exception("Default R not feasible: M != N")
            else:
                self.R = np.identity(self.N)

        self.distributions = [('piA', 'gaussian', self.K, 0)]
        if init == 'random':
            self.variational_params = [-1 * np.ones(self.K), -5 * np.ones(self.K)]
        elif init == 'kmeans' or init == 'gmm':
            label_counter = Counter(self.zA)
            self.variational_params = [np.array([label_counter[label] / float(self.N) for label in range(self.K)]), -5 * np.ones(self.K)]

        self.distributions.append(('piB', 'gaussian', self.K, self.K*2))
        if init == 'random':
            self.variational_params.extend([-1 * np.ones(self.K), -5 * np.ones(self.K)])
        elif init == 'kmeans' or init == 'gmm':
            label_counter = Counter(self.zB)
            self.variational_params.extend([np.array([label_counter[label] / float(self.M) for label in range(self.K)]), -5 * np.ones(self.K)])

        if self.phi is None:
            self.distributions.append(('phi', 'gaussian', 1, self.K*4))
            self.variational_params.extend([np.array(0.), np.array(10.)])

        self.params = np.hstack(self.variational_params)

        self.adjust_cluster_relationships()

    def total_sample_density(self, param_samples, t):
        pass

    def total_density(self, params, t):
        pass

    def realign(self):
        new_zB = deepcopy(self.zB)
        new_muB = deepcopy(self.muB)
        new_sigmaB = deepcopy(self.sigmaB)

        for k in range(self.K):
            related_l = self.get_related_cluster(0, k)
            new_zB = np.where(self.zB == related_l, k, new_zB)
            new_muB[k] = self.muB[related_l]
            new_sigmaB[k] = self.sigmaB[related_l]

        self.muB = new_muB
        self.sigmaB = new_sigmaB
        self.zB = new_zB
        self.C = np.identity(self.C.shape[0])

    def plot_R(self):
        sns.set()
        matrix = self.R
        ax = sns.heatmap(matrix, cmap='Blues', cbar=True, xticklabels=True, yticklabels=True, vmin=0)
        plt.title('Current Iteration Affinity')
        plt.show()

    def plot_C(self):
        sns.set()
        matrix = self.C
        ax = sns.heatmap(matrix, cmap='Blues', cbar=True, xticklabels=True, yticklabels=True, vmin=0)
        plt.title('Current Iteration Affinity')
        plt.show()

    def get_clustering(self):
        clustersA = {}
        clustersB = {}
        for k in range(self.K):
            entity_indices_A = np.where(self.zA == k)
            entity_indices_B = np.where(self.zB == k)
            clustersA[k] = entity_indices_A
            clustersB[k] = entity_indices_B
        return clustersA, clustersB

    def get_related_cluster(self, view, cluster):
        if view == 0:
            return np.nonzero(self.C[cluster])[0][0]
        else:
            return np.nonzero(self.C[:, cluster])[0][0]

    def are_clusters_related(self, cluster1, cluster2):
        return self.C[cluster1, cluster2] == 1

    def get_related_indices(self, query_index, view):
        if view == 1:
            cc = self.ccs[query_index]
        elif view == 2:
            cc = self.ccs[query_index + self.N]
        indices = np.where(self.ccs == cc)[0]
        view1_indices = [index for index in indices if index < self.N]
        view2_indices = [index - self.N for index in indices if index >= self.N]
        if view == 1:
            view1_indices.remove(query_index)
            return view1_indices, view2_indices
        elif view == 2:
            view2_indices.remove(query_index)
            return view2_indices, view1_indices

    def count_num_fused(self):
        num_fused = 0
        for i in range(self.N):
            related_entities = self.R_dict[i]
            for j in related_entities:
                if self.C[self.zA[i], self.zB[j]] == 1:
                    num_fused += 1
        self.num_fused = num_fused
        return num_fused

    def count_num_grouped(self, view):
        component_labels = np.unique(self.ccs)
        num_grouped = 0
        z = self.zA if view == 0 else self.zB
        for label in component_labels:
            indices = np.where(self.ccs == label)[0]
            view1_indices = [index for index in indices if index < self.N]
            view2_indices = [index - self.N for index in indices if index >= self.N]
            indices = view1_indices if view == 0 else view2_indices
            if len(indices) == 0:
                continue
            score = 1 / float(len(set(z[indices])))
            # print(z[indices], score)
            num_grouped += score
        return num_grouped

    def plot_relatedness(self):
        sns.set()
        matrix = compute_affinity(self.zA, self.zB, self.R, self.C, self.K, self.K)
        ax = sns.heatmap(matrix, cmap='Blues', cbar=True, xticklabels=True, yticklabels=True, vmin=0)
        plt.title('Current Iteration Affinity')
        plt.show()

    def compute_component_joint_modifier(self, view1_indices, view2_indices, view1_assignments, view2_assignments, phi_mod):  # TODO: merge with other method
        modifier = 1
        if self.phi != 0:
            for index1 in range(len(view1_indices)):
                for index2 in range(len(view2_indices)):
                    if self.C[view1_assignments[index1], view2_assignments[index2]] == 1 and self.R[view1_indices[index1], view2_indices[index2]]:
                        modifier = modifier * phi_mod
        return modifier

    def get_num_related_between_clusters(self, view1, cluster1, view2, cluster2):
        # checked, correct
        related_entities_in_view2 = self.get_related_entities_for_cluster(view1, cluster1)
        cluster2_indices = list(np.where(self.z_list[view2] == cluster2)[0])
        fused = [entity for entity in related_entities_in_view2 if entity in cluster2_indices]
        return len(fused)

    def get_related_entities_for_cluster(self, view, cluster_label):
        # checked, correct
        cur_z = self.z_list[view]
        cur_indices = list(np.where(cur_z == cluster_label)[0])
        rel_indices = []
        for sample in cur_indices:
            if view == 0:
                cur_match = list(np.where(self.R[sample, :] == 1)[0])
            elif view == 1:
                cur_match = list(np.where(self.R[:, sample] == 1)[0])
            if cur_match is not None:
                rel_indices.extend(cur_match)
        return rel_indices

    def get_fused_entities_for_cluster(self, view, cluster_label):
        fused_cluster = self.get_related_cluster(view, cluster_label)
        other_view = 1 if view == 0 else 0
        other_z = self.z_list[other_view]
        related_entities = self.get_related_entities_for_cluster(view, cluster_label)
        num_fused = np.count_nonzero(other_z[related_entities] == fused_cluster)
        return num_fused

    def adjust_cluster_relationships(self):
        # self.plot_relatedness()
        for view in range(len(self.z_list) - 1):
            other_view = 1 if view == 0 else 0
            cur_z = self.zA if view == 0 else self.zB
            clusters = Counter(cur_z)
            for cur_cluster, size in clusters.items():
                fused_entities = self.get_fused_entities_for_cluster(view, cur_cluster)
                cur_total_fused = 0
                old_relationship = self.get_related_cluster(view, cur_cluster)
                new_relationship = old_relationship
                swap_cluster = cur_cluster
                for candidate_cluster in range(self.K):
                    candidate_relationship = self.get_related_cluster(view, candidate_cluster)
                    candidate_fused_entities = self.get_fused_entities_for_cluster(view, candidate_cluster)
                    new_fused_entities = self.get_num_related_between_clusters(view, cur_cluster, other_view, candidate_relationship)
                    new_candidate_fused_entities = self.get_num_related_between_clusters(view, candidate_cluster, other_view, old_relationship)
                    total_fused = fused_entities + candidate_fused_entities
                    new_total_fused = new_fused_entities + new_candidate_fused_entities
                    if new_total_fused > total_fused and new_total_fused > cur_total_fused:
                        new_relationship = candidate_relationship
                        swap_cluster = candidate_cluster
                        cur_total_fused = new_total_fused
                if old_relationship == new_relationship:
                    continue
                # import pdb; pdb.set_trace()
                if view == 0:
                    print("Adjusting relationships...", cur_cluster, swap_cluster)
                    self.C[cur_cluster, new_relationship] = 1
                    self.C[cur_cluster, old_relationship] = 0

                    self.C[swap_cluster, old_relationship] = 1
                    self.C[swap_cluster, new_relationship] = 0

                elif view == 1:
                    print("Adjusting relationships...", cur_cluster, swap_cluster)
                    self.C[new_relationship, cur_cluster] = 1
                    self.C[old_relationship, cur_cluster] = 0

                    self.C[old_relationship, swap_cluster] = 1
                    self.C[new_relationship, swap_cluster] = 0

        print("Adjusted labeling:", Counter(self.zA), Counter(self.zB))
        # self.plot_C()

    def run_connected_components(self):
        x = np.concatenate((np.zeros((self.N, self.N)), self.R), 1)
        y = np.concatenate((self.R.T, np.zeros((self.M, self.M))), 1)
        g = np.concatenate((x, y), 0)
        num_components, labels = connected_components(g)
        related_dict = defaultdict(list)
        for component in np.unique(labels):
            indices = np.where(labels == component)[0]
            view1_indices = [index for index in indices if index < self.N]
            view2_indices = [index - self.N for index in indices if index >= self.N]
            for index1 in view1_indices:
                for index2 in view2_indices:
                    if self.R[index1, index2]:
                        related_dict[component].append((index1, index2))
        return labels, related_dict

    def estimate_cluster_parameters(self, view):
        if view == 1:
            labels = self.zA
            data = self.view1
            old_mus = self.muA
            old_sigmas = self.sigmaA
        else:
            labels = self.zB
            data = self.view2
            old_mus = self.muB
            old_sigmas = self.sigmaB
        mus = []
        sigmas = []
        for i in range(self.K):
            cluster_samples = data[labels == i, :]
            if not cluster_samples.shape[0] == 0:
                mu = np.mean(cluster_samples, 0)
                sigma = np.cov(cluster_samples.T, bias=True)
                dim = len(mu)
                num_samples = len(cluster_samples)
                posterior_v = dim + self.inv_wish_v + num_samples
                posterior_sigma = num_samples * sigma + self.inv_wish_psi * np.identity(dim)
                posterior_sigma_est = posterior_sigma / (posterior_v - dim - 1)
                mus.append(mu)
                sigmas.append(posterior_sigma_est)
            else:
                mus.append(old_mus[i])
                sigmas.append(old_sigmas[i])
        return mus, sigmas

    def phi_marginal(self, phi):
        # return gamma.logpdf(phi / 0.2 + 0.000001, a=1) - np.log(0.2)
        return norm.logpdf(phi, self.phi_mean_prior, self.phi_scale_prior)

    def cluster_density(self, params, t):
        likelihood = 0

        if self.phi is None:
            phi = params[-2]
        else:
            phi = float(self.phi)

        phi_mod = (1 + phi)**self.phi_power

        phi_marginal = self.phi_marginal(phi)
        likelihood = likelihood + phi_marginal

        #################################################
        ############### Cluster priors ##################
        #################################################

        prior_muA = np.sum(invwishart.logpdf(np.array(self.sigmaA).T, df=self.inv_wish_v + self.D1, scale=self.inv_wish_psi * np.identity(self.D1)))
        prior_muB = np.sum(invwishart.logpdf(np.array(self.sigmaB).T, df=self.inv_wish_v + self.D2, scale=self.inv_wish_psi * np.identity(self.D2)))
        likelihood = likelihood + prior_muA + prior_muB

        #################################################
        #################### View A #####################
        #################################################

        piA = regularize(softplus(params[:self.K]))
        piA = piA / piA.sum()

        alphas = np.array([self.alpha_A / self.K for i in range(self.K)])
        likelihood = likelihood + stats.dirichlet.logpdf(piA, alphas)

        cluster_samples = defaultdict(list)
        for sample_index in range(len(self.zA)):
            cluster_samples[self.zA[sample_index]].append(sample_index)

        for cluster in range(self.K):
            cluster_likelihood = 0
            sample_indices = cluster_samples[cluster]
            # cluster_likelihood = cluster_likelihood + np.log(piA[cluster]) * len(sample_indices)
            cluster_likelihood = cluster_likelihood + np.sum(mvn.logpdf(self.view1[sample_indices, :], self.muA[cluster], self.sigmaA[cluster]))
            likelihood = likelihood + cluster_likelihood

        #################################################
        #################### View B #####################
        #################################################

        piB = regularize(softplus(params[self.K * 2:self.K * 3]))
        piB = piB / piB.sum()

        alphas = np.array([self.alpha_B / self.K for i in range(self.K)])
        likelihood = likelihood + stats.dirichlet.logpdf(piB, alphas)

        cluster_samples = defaultdict(list)
        for sample_index in range(len(self.zB)):
            cluster_samples[self.zB[sample_index]].append(sample_index)

        for cluster in range(self.K):
            cluster_likelihood = 0
            sample_indices = cluster_samples[cluster]
            # cluster_likelihood = cluster_likelihood + np.log(piB[cluster]) * len(sample_indices)
            cluster_likelihood = cluster_likelihood + np.sum(mvn.logpdf(self.view2[sample_indices, :], self.muB[cluster], self.sigmaB[cluster]))
            likelihood = likelihood + cluster_likelihood

        return piA, piB, phi_mod, likelihood

    def cluster_sample_density(self, param_samples, t):
        likelihood = 0

        if self.phi is None:
            phi = hardplus(param_samples[:, -1])
        else:
            phi = float(self.phi)

        phi_mod = (1 + phi)**self.phi_power

        phi_marginal = self.phi_marginal(phi)
        likelihood = likelihood + phi_marginal

        #################################################
        ############### Cluster priors ##################
        #################################################

        prior_muA = np.sum(invwishart.logpdf(np.array(self.sigmaA).T, df=self.inv_wish_v + self.D1, scale=self.inv_wish_psi * np.identity(self.D1)))
        prior_muB = np.sum(invwishart.logpdf(np.array(self.sigmaB).T, df=self.inv_wish_v + self.D2, scale=self.inv_wish_psi * np.identity(self.D2)))
        likelihood = likelihood + prior_muA + prior_muB

        #################################################
        #################### View A #####################
        #################################################

        piA = regularize(softplus(param_samples[:, 0:self.K]))
        piA = (piA.T / np.sum(piA, axis=1)).T

        alphas = np.array([self.alpha_A / self.K for i in range(self.K)])
        likelihood = likelihood + np.array([stats.dirichlet.logpdf(sample, alphas) for sample in piA])

        cluster_samples = defaultdict(list)
        for sample_index in range(len(self.zA)):
            cluster_samples[self.zA[sample_index]].append(sample_index)

        for cluster in range(self.K):
            cluster_likelihood = 0
            sample_indices = cluster_samples[cluster]
            # cluster_likelihood = np.array([np.log(piA[i, cluster]) * len(sample_indices) for i in range(len(param_samples))])
            cluster_likelihood = cluster_likelihood + np.sum(mvn.logpdf(self.view1[sample_indices, :], self.muA[cluster], self.sigmaA[cluster]))
            likelihood = likelihood + cluster_likelihood

        #################################################
        #################### View B #####################
        #################################################
        piB = regularize(softplus(param_samples[:, self.K:self.K * 2]))
        piB = (piB.T / np.sum(piB, axis=1)).T

        alphas = np.array([self.alpha_B / self.K for i in range(self.K)])
        likelihood = likelihood + np.array([stats.dirichlet.logpdf(sample, alphas) for sample in piB])

        cluster_samples = defaultdict(list)
        for sample_index in range(len(self.zB)):
            cluster_samples[self.zB[sample_index]].append(sample_index)

        for cluster in range(self.K):
            cluster_likelihood = 0
            sample_indices = cluster_samples[cluster]
            # cluster_likelihood = np.array([np.log(piB[i, cluster]) * len(sample_indices) for i in range(len(param_samples))])
            cluster_likelihood = cluster_likelihood + np.sum(mvn.logpdf(self.view2[sample_indices, :], self.muB[cluster], self.sigmaB[cluster]))
            likelihood = likelihood + cluster_likelihood

        return piA, piB, phi_mod, np.array(likelihood)

    def compute_log_component_joint(self, view1_indices, view2_indices, view1_assignments, view2_assignments, piA, piB, phi_mod):
        component_joint = np.sum(np.log(piA[:, view1_assignments]), 1) + np.sum(np.log(piB[:, view2_assignments]), 1)
        if self.phi != 0:
            for index1 in range(len(view1_indices)):
                for index2 in range(len(view2_indices)):
                    if self.C[view1_assignments[index1], view2_assignments[index2]] == 1 and self.R[view1_indices[index1], view2_indices[index2]]:
                        component_joint = component_joint + np.log(phi_mod)
        return component_joint

    def log_var(self, param_samples, params):
        log_prob = mvn.logpdf(param_samples[:, 0:self.K], params[0:self.K], np.diag(regularize(softplus(params[self.K:self.K * 2]))))
        log_prob = log_prob + mvn.logpdf(param_samples[:, self.K:self.K * 2], params[self.K * 2:self.K * 3], np.diag(regularize(softplus(params[self.K * 3:self.K * 4]))))
        if self.phi is None:
            log_prob = log_prob + norm.logpdf(param_samples[:, -1], params[-2], np.sqrt(regularize(softplus(params[-1]))))  # phi
        return log_prob

    def generate_cluster_probabilities(self, piA, piB):
        cluster_probabilitiesA = []
        cluster_probabilitiesB = []
        for cluster in range(self.K):
            probabilitiesA = mvn.logpdf(self.view1, self.muA[cluster], regularize(self.sigmaA[cluster], diagonal=True))
            probabilitiesB = mvn.logpdf(self.view2, self.muB[cluster], regularize(self.sigmaB[cluster], diagonal=True))
            cluster_probabilitiesB.append(probabilitiesB)
            cluster_probabilitiesA.append(probabilitiesA)

        print("piA:", piA)
        print("piB:", piB)

        total_probabilitiesA = np.log(piA) + np.asarray(cluster_probabilitiesA).T
        total_probabilitiesB = np.log(piB) + np.asarray(cluster_probabilitiesB).T
        return total_probabilitiesA, total_probabilitiesB

    def update(self, params):
        new_params = deepcopy(params)
        new_params[-2] = hardplus(new_params[-2])
        return new_params

    def serial_assignment_maximization(self, component_labels, sample_probabilitiesA, sample_probabilitiesB, phi_mod):
        total_likelihood = 0
        for label in component_labels:
            indices = np.where(self.ccs == label)[0]
            view1_indices = [index for index in indices if index < self.N]
            view2_indices = [index - self.N for index in indices if index >= self.N]
            view1_combos = list(product(range(self.K), repeat=len(view1_indices)))
            view2_combos = list(product(range(self.K), repeat=len(view2_indices)))
            best_combos, max_likelihood = self.test_combos(view1_combos, view2_combos, view1_indices, view2_indices, sample_probabilitiesA, sample_probabilitiesB, phi_mod)
            total_likelihood = total_likelihood + max_likelihood
            np.put(self.zA, view1_indices, best_combos[0])
            np.put(self.zB, view2_indices, best_combos[1])
        # print("total helper likelihood:", total_likelihood)

    def alternating_assignment(self, view, cluster_probabilities, cur_assign, other_assign, phi_mod):
        new_assign = deepcopy(cur_assign)
        component_labels = np.unique(self.ccs)
        for label in component_labels:
            indices = np.where(self.ccs == label)[0]
            if view == 1:
                curview_indices = [index for index in indices if index < self.N]
            elif view == 2:
                curview_indices = [index - self.N for index in indices if index >= self.N]
            for index1 in curview_indices:
                cur_best_prob = -9999999999
                cur_best_assign = -1
                related = np.nonzero(self.R[index1, :])[0] if view == 1 else np.nonzero(self.R[:, index1])[0]
                C = self.C if view == 1 else self.C.T
                for assignment in range(self.K):
                    num_fused = len([entity for entity in related if C[assignment, other_assign[entity]] == 1])
                    prob = cluster_probabilities[index1, assignment]
                    prob = prob + num_fused * np.log(phi_mod)
                    if prob < -9999999999:
                        import pdb; pdb.set_trace()
                    if prob > cur_best_prob:
                        cur_best_prob = prob
                        cur_best_assign = assignment
                new_assign[index1] = cur_best_assign
        if view == 1:
            self.zA = new_assign
        elif view == 2:
            self.zB = new_assign

    def reestimate_assignments(self, piA, piB, phi_mod):
        component_labels = np.unique(self.ccs)
        total_probabilitiesA, total_probabilitiesB = self.generate_cluster_probabilities(piA, piB)
        print("Reassigning...")
        if phi_mod == 1:
            # below only valid if phi is 0!!!
            argmax_zA = np.argmax(total_probabilitiesA, axis=1)
            argmax_zB = np.argmax(total_probabilitiesB, axis=1)
            print(Counter(argmax_zA), Counter(argmax_zB))
            self.zA = argmax_zA
            self.zB = argmax_zB
        else:
            if self.inference == 'serial':
                self.serial_assignment_maximization(component_labels, total_probabilitiesA, total_probabilitiesB, phi_mod)
            elif self.inference == 'alternating':
                self.alternating_assignment(1, total_probabilitiesA, self.zA, self.zB, phi_mod)
                self.alternating_assignment(2, total_probabilitiesB, self.zB, self.zA, phi_mod)
            print(Counter(self.zA), Counter(self.zB))

        self.z_list = [self.zA, self.zB]

    def test_combos(self, view1_combos, view2_combos, view1_indices, view2_indices, sampleA, sampleB, phi_mod):
        max_likelihood = -9999999999
        best_combos = None
        for combo1 in view1_combos:
            for combo2 in view2_combos:
                combo_likelihood = 0
                for index1 in range(len(view1_indices)):
                    combo_likelihood += sampleA[view1_indices[index1], combo1[index1]]
                    for index2 in range(len(view2_indices)):
                        if self.C[combo1[index1], combo2[index2]] == 1 and self.R[view1_indices[index1], view2_indices[index2]]:
                            combo_likelihood = combo_likelihood + np.log(phi_mod)
                for index2 in range(len(view2_indices)):
                    combo_likelihood += sampleB[view2_indices[index2], combo2[index2]]
                if max_likelihood < combo_likelihood:
                    max_likelihood = combo_likelihood
                    best_combos = combo1, combo2
        return best_combos, max_likelihood

    def get_z(self):
        return self.zA, self.zB
