from __future__ import absolute_import
# from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from copy import copy
from autograd import grad
from autograd.misc.optimizers import adam
from helpers import *


def gd(objective, random_state=None):

    rs = npr.RandomState(random_state)

    def regularized_objective(params, t):
        return -1 * objective(params, t)

    gradient = grad(regularized_objective)

    return regularized_objective, gradient


def run_gd(model, step_size=1, num_iters=1000, plottable=False, random_state=None):

    objective, gradient = \
        gd(model.total_density, random_state=random_state)

    # Set up figure.
    if plottable:
        print("PLOTTING IS ON")
        fig = plt.figure(figsize=(8, 8), facecolor='white')
        ax = fig.add_subplot(111, frameon=False)
        plt.ion()
        plt.show(block=False)

    def callback(params, t, g):
        print("Iteration {} objective {}".format(t, objective(params, t)))
        model.callback(params, t, g)

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
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
        x = model.update(x)
    return x
