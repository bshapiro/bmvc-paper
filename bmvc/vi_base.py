from autograd import grad
from autograd.misc import flatten
from helpers import plot_isocontours, regularize, softplus
import autograd.builtins as btn
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import matplotlib.pyplot as plt
from copy import copy


class Model(object):
    """
    Base model class. Sets up parameter handling.
    Every new model should extend the model class and override
    the log_density method.
    """

    def __init__(self, data=None):
        """
        This method should be overriden. Use it to define
        self.var_initializations and self.param_names, then
        call self.setup(data) to finish model setup. Optionally,
        define discrete param names and discrete initializations
        as well.
        """
        self.data = data

    def log_density(self, param_samples, t):
        """
        This method should be overriden. It should return
        the joint log density of the data and each set of parameter
        samples using the names defined in the param samples in param_samples.
        """
        pass

    def log_var(self, param_samples, variational_params):
        """
        This method should be overridden if the variational approximation
        does not have an analytic entropy. It should return the log density
        of the param samples according to the variational parameters (e.g.
        log q(z | lambda)).
        """
        pass

    def callback(self):
        """
        Use this method to execute any changes to the model that need to
        be accomplished between iterations, for example to update discrete
        variables.
        """
        pass


def black_box_variational_inference(logprob, logvar, distributions, num_samples, random_state=None):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        return flatten(params)[0]

    def gaussian_entropy(log_std):
        return 0.5 * len(log_std) * (1.0 + np.log(2 * np.pi)) + np.sum(log_std)

    rs = npr.RandomState(random_state)

    def variational_objective(params, t):
        variational_params = unpack_params(params)
        samples = btn.list([])
        for i in range(len(distributions)):
            param_name, distribution, dimension, param_index = distributions[i]
            if distribution == 'gaussian':
                mean = np.array(variational_params[param_index:param_index + dimension])
                cov = regularize(softplus(np.array(variational_params[param_index + dimension:param_index + dimension * 2])))
                noise_samples = np.array(rs.randn(num_samples, len(mean)))
                cur_samples = noise_samples * np.sqrt(cov) + mean
                samples.append(cur_samples)
        total_samples = np.hstack(samples)
        if len(distributions) == 1 and distribution == 'gaussian':
            lower_bound = gaussian_entropy(np.log(np.sqrt(cov))) + np.mean(logprob(total_samples, t))
        else:
            # lower_bound = np.mean(logprob(total_samples, t))
            lower_bound = np.mean(-logvar(total_samples, variational_params) + logprob(total_samples, t))
        # print(params[-2:])
        return -lower_bound  # make it negative so that we can do gradient descent

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params


def run_vi(model, step_size=1, num_samples=10, num_iters=1000, plottable=False, random_state=None):

    objective, gradient, unpack_params = \
        black_box_variational_inference(model.total_sample_density, model.log_var, model.distributions, num_samples=num_samples, random_state=random_state)

    # Set up figure.
    if plottable:
        print("PLOTTING IS ON")
        fig = plt.figure(figsize=(8, 8), facecolor='white')
        ax = fig.add_subplot(111, frameon=False)
        plt.ion()
        plt.show(block=False)

    def callback(params, t, g):
        # print("Iteration {} lower bound {}".format(t, -objective(params, t)))
        print("Iteration {} of VI complete.".format(t))
        params = unpack_params(params)
        model.callback(params, t, g)

        if plottable:
            mean, std = params[0:2], params[2:4]
            print(mean, std)

            plt.cla()
            target_distribution = lambda x: np.exp(objective(x, t))
            plot_isocontours(ax, target_distribution)

            variational_contour = lambda x: mvn.pdf(x, mean, np.diag(2 * softplus(std)))
            plot_isocontours(ax, variational_contour)
            plt.draw()
            plt.pause(1.0 / 30.0)

        # visualize_2d_results(model.view1, model.view2, model.zA, model.zB, model.muA, model.muB, model.sigmaA, model.sigmaB)

    print("Optimizing variational parameters...")
    x = copy(model.params)
    b1 = 0.9
    b2 = 0.999
    eps = 10**-8
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = gradient(x, i)
        if callback:
            callback(x, i, g)
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size * mhat / (np.sqrt(vhat) + eps)
        x = model.update(x)

    return x
