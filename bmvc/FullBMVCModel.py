from BaseBMVCModel import BaseBMVCModel
import autograd.numpy as np
from collections import Counter
from itertools import product
from helpers import softplus, regularize


class FullBMVCModel(BaseBMVCModel):

    def __init__(self,
                 view1,
                 view2,
                 R=None,
                 K=5,
                 phi=None,
                 phi_mean_prior=0,
                 phi_scale_prior=0.1,
                 init='kmeans',
                 random_state=None,
                 inference='alternating',
                 phi_power=1,
                 track_convergence=False):
        super(FullBMVCModel, self).__init__(view1,
                                           view2,
                                           R=R,
                                           K=K,
                                           phi=phi,
                                           phi_mean_prior=phi_mean_prior,
                                           phi_scale_prior=phi_scale_prior,
                                           init=init,
                                           random_state=random_state,
                                           inference=inference,
                                           phi_power=phi_power)
        self.likelihoods = []
        self.pseudolikelihoods = []
        self.track_convergence = track_convergence

        if self.track_convergence:
            from PseudoBMVCModel import PseudoBMVCModel
            self.pseudo_model = PseudoBMVCModel(view1,
                                               view2,
                                               R=R,
                                               K=K,
                                               phi=phi,
                                               phi_mean_prior=phi_mean_prior,
                                               phi_scale_prior=phi_scale_prior,
                                               init=init,
                                               random_state=random_state,
                                               inference=inference,
                                               phi_power=phi_power)

    def total_sample_density(self, param_samples, t):
        piA, piB, phi_mod, likelihood = self.cluster_sample_density(param_samples, t)
        z_joint = self.compute_z_joint(piA, piB, phi_mod)
        likelihood = likelihood + z_joint
        return likelihood

    def total_density(self, params, t):
        piA, piB, phi_mod, likelihood = self.cluster_density(params, t)
        z_joint = self.compute_z_joint(piA, piB, phi_mod)
        likelihood = likelihood + z_joint
        return likelihood

    def compute_z_joint(self, piA, piB, phi_mod):
        z_joint = 0
        component_labels = np.unique(self.ccs)
        n_samples = 1 if piA.ndim == 1 else piA.shape[0]

        for label in component_labels:
            indices = np.where(self.ccs == label)[0]
            view1_indices = [index for index in indices if index < self.N]
            view2_indices = [index - self.N for index in indices if index >= self.N]
            view1_assignments = self.zA[view1_indices]
            view2_assignments = self.zB[view2_indices]
            component_joint = self.compute_log_component_joint(view1_indices, view2_indices, view1_assignments, view2_assignments, piA.reshape((n_samples, self.K)), piB.reshape((n_samples, self.K)), phi_mod)
            z_joint = z_joint + component_joint

        #################################################
        ########## Normalization constants ##############
        #################################################
        if self.phi != 0:
            z_joint = z_joint - self.log_norm_c(piA.reshape((n_samples, self.K)), piB.reshape((n_samples, self.K)), phi_mod)
        return z_joint

    def log_norm_c(self, piA, piB, phi_mod):
        # print("Computing normalization constant...")
        component_labels = np.unique(self.ccs)
        total_c = 0
        for label in component_labels:
            indices = np.where(self.ccs == label)[0]
            view1_indices = [index for index in indices if index < self.N]
            view2_indices = [index - self.N for index in indices if index >= self.N]
            c = self.component_norm_c(view1_indices, view2_indices, label, piA, piB, phi_mod)
            total_c = total_c + np.log(c)
        return total_c

    def component_norm_c(self, view1_indices, view2_indices, component_label, piA, piB, phi_mod):
        view1_combos = list(product(range(self.K), repeat=len(view1_indices)))
        view2_combos = list(product(range(self.K), repeat=len(view2_indices)))
        total_likelihood = 0

        for combo1 in view1_combos:
            for combo2 in view2_combos:
                combo_likelihood = np.prod(piA[:, combo1], 1)
                combo_likelihood = combo_likelihood * np.prod(piB[:, combo2], 1)
                modifier = self.compute_component_joint_modifier(view1_indices, view2_indices, combo1, combo2, phi_mod)
                total_likelihood = total_likelihood + combo_likelihood * modifier
        return total_likelihood

    def callback(self, params, t, g):
        if self.prev_params is None:
            self.prev_params = params
        else:
            print("PARAM DIFF:", params - self.prev_params)
            self.prev_params = params
        print("GRADIENT:", g)
        # print("PARAMS:", params)

        print("NUM FUSED: ", self.count_num_fused())
        print("NUM GROUPED, VIEW 1: ", self.count_num_grouped(0))
        print("NUM GROUPED, VIEW 2: ", self.count_num_grouped(1))

        if self.phi is None:
            phi = params[-2]
        else:
            phi = float(self.phi)

        phi_mod = (1 + phi)**self.phi_power

        print("PHI", phi, "PHI MOD", phi_mod)
        self.prev_phi = phi_mod

        log_likelihood_M2 = self.total_density(params, t)
        print("Total log likelihood M2: ", log_likelihood_M2)

        if log_likelihood_M2 < self.prev_log_likelihood_A:
            print("M2 NON-INCREASING")
        self.prev_log_likelihood_M2 = log_likelihood_M2

        piA = regularize(softplus(params[:self.K]))
        piA = piA / piA.sum()

        piB = regularize(softplus(params[self.K * 2:self.K * 3]))
        piB = piB / piB.sum()

        self.reestimate_assignments(piA, piB, phi_mod)

        log_likelihood_E = self.total_density(params, t)
        print("Total log likelihood, E: ", log_likelihood_E)
        if log_likelihood_E < self.prev_log_likelihood_M2:
            print("Likelihood E NON-INCREASING")
        self.prev_log_likelihood_E = log_likelihood_E

        self.muA, self.sigmaA = self.estimate_cluster_parameters(1)
        self.muB, self.sigmaB = self.estimate_cluster_parameters(2)

        log_likelihood_M1 = self.total_density(params, t)
        print("Total log likelihood, M.1: ", log_likelihood_M1)
        if log_likelihood_M1 < self.prev_log_likelihood_E:
            print("Likelihood M1 NON-INCREASING")
        self.prev_log_likelihood_M1 = log_likelihood_M1

        self.adjust_cluster_relationships()

        log_likelihood_A = self.total_density(params, t)
        print("Total log likelihood, A: ", log_likelihood_A)
        if log_likelihood_A < self.prev_log_likelihood_M1:
            print("Likelihood A NON-INCREASING")
        if abs(self.prev_log_likelihood_A - log_likelihood_A) < 0.1:
            print("Converged!")
        self.prev_log_likelihood_A = log_likelihood_A

        if self.track_convergence:
            self.pseudo_model.muA = self.muA
            self.pseudo_model.muB = self.muB
            self.pseudo_model.sigmaA = self.sigmaA
            self.pseudo_model.sigmaB = self.sigmaB
            self.pseudo_model.zA = self.zA
            self.pseudo_model.zB = self.zB
            self.pseudo_model.C = self.C
            pseudo = self.pseudo_model.calculate_pseudolikelihood(params, t)
            full = self.calculate_full_likelihood(params, t)
            self.pseudolikelihoods.append(pseudo)
            self.likelihoods.append(full)

    def calculate_full_likelihood(self, params, t):
        prev_pseudo = self.pseudo
        self.pseudo = False
        likelihood = self.total_density(params, t)
        self.pseudo = prev_pseudo
        return likelihood
