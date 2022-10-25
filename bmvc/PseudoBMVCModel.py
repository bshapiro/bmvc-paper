from BaseBMVCModel import BaseBMVCModel
import autograd.numpy as np
from helpers import softplus, regularize


class PseudoBMVCModel(BaseBMVCModel):

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
        super(PseudoBMVCModel, self).__init__(view1,
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
            from FullBMVCModel import FullBMVCModel
            self.full_model = FullBMVCModel(view1,
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
            component_size = len(indices)
            view1_indices = [index for index in indices if index < self.N]
            view2_indices = [index - self.N for index in indices if index >= self.N]
            view1_assignments = self.zA[view1_indices]
            view2_assignments = self.zB[view2_indices]
            component_joint = self.compute_log_component_joint(view1_indices, view2_indices, view1_assignments, view2_assignments, piA.reshape((n_samples, self.K)), piB.reshape((n_samples, self.K)), phi_mod)
            if self.phi == 0:
                z_joint = z_joint + component_joint
            else:
                z_joint = z_joint + component_joint * component_size

        #################################################
        ########## Normalization constants ##############
        #################################################

        if self.phi != 0:
            z_joint = z_joint - self.pseudo_log_norm_c(piA.reshape((n_samples, self.K)), piB.reshape((n_samples, self.K)), phi_mod)

        return z_joint

    def pseudo_log_norm_c(self, piA, piB, phi_mod):
        component_labels = np.unique(self.ccs)
        total_c = 0
        for label in component_labels:
            indices = np.where(self.ccs == label)[0]
            view1_indices = [index for index in indices if index < self.N]
            view2_indices = [index - self.N for index in indices if index >= self.N]
            c = self.pseudo_component_norm_c(view1_indices, view2_indices, piA, piB, phi_mod)
            total_c = total_c + c
        return total_c

    def pseudo_component_norm_c(self, view1_indices, view2_indices, piA, piB, phi_mod):
        view1_assignments = self.zA[view1_indices]
        view2_assignments = self.zB[view2_indices]  # TODO: see if array casting here makes a difference
        view1_range = list(range(len(view1_indices)))
        view2_range = list(range(len(view2_indices)))

        component_norm = 0

        for sample in view1_range:
            component_assignment_norm1 = 0
            pre_sample = view1_assignments[:sample]
            post_sample = view1_assignments[sample+1:]

            for k in range(self.K):
                new_view1_assignments = np.concatenate((pre_sample, np.array(k).reshape(1), post_sample), 0)
                new_view1_piprod = np.prod(np.array(piA[:, new_view1_assignments]), 1)
                view2_piprod = np.prod(np.array(piB[:, view2_assignments]), 1)
                modifier1 = self.compute_component_joint_modifier(view1_indices, view2_indices, new_view1_assignments, view2_assignments, phi_mod)
                component_assignment_norm1 = component_assignment_norm1 + new_view1_piprod * view2_piprod * modifier1
            component_norm = component_norm + np.log(component_assignment_norm1)

        for sample in view2_range:
            component_assignment_norm2 = 0
            pre_sample = view2_assignments[:sample]
            post_sample = view2_assignments[sample+1:]

            for k in range(self.K):
                new_view2_assignments = np.concatenate((pre_sample, np.array(k).reshape(1), post_sample), 0)
                new_view2_piprod = np.prod(np.array(piB[:, new_view2_assignments]), 1)
                view1_piprod = np.prod(np.array(piA[:, view1_assignments]), 1)
                modifier2 = self.compute_component_joint_modifier(view1_indices, view2_indices, view1_assignments, new_view2_assignments, phi_mod)
                component_assignment_norm2 = component_assignment_norm2 + view1_piprod * new_view2_piprod * modifier2
            component_norm = component_norm + np.log(component_assignment_norm2)

        return component_norm

    def calculate_pseudolikelihood(self, params, t):
        likelihood = self.total_density(params, t)
        return likelihood

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
        if abs(self.prev_log_likelihood_A - log_likelihood_A) < 0.01:
            print("Converged!")
            self.converged = True
        self.prev_log_likelihood_A = log_likelihood_A

        if self.track_convergence:
            self.full_model.muA = self.muA
            self.full_model.muB = self.muB
            self.full_model.sigmaA = self.sigmaA
            self.full_model.sigmaB = self.sigmaB
            self.full_model.zA = self.zA
            self.full_model.zB = self.zB
            self.full_model.C = self.C
            pseudo = self.calculate_pseudolikelihood(params, t)
            full = self.full_model.calculate_full_likelihood(params, t)
            self.pseudolikelihoods.append(pseudo)
            self.likelihoods.append(full)
