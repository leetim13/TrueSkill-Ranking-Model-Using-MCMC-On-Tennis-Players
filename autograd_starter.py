from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.misc.optimizers import adam


def unpack_params(params):
    D = params.size // 2
    # Variational dist is a diagonal Gaussian.
    mean, log_std = params[:D], params[D:]
    return mean, log_std

def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

rs = npr.RandomState(0)

def elbo(params, t, logprob, D, num_samples):
    mean, log_std = #TODO
    samples = #TODO
    logp = #TODO
    logq = #TODO
    return #TODO


def logp_a_beats_b(za, zb):
    # za and zb are (num_samples x 1)
    return #TODO

def log_prior(zs):
    # zs is (num_samples x num_players)
    return #TODO

def all_matches_likelihood(matches, zs):
    # matches is an array of size (num_games x 2)
    # zs is an array of size (num_samples x num_players)
    zs_a = #TODO
    zs_b = #TODO
    likes = #TODO
    return #TODO

# Set up plotting code
def plot_isocontours(ax, func, xlimits=[-3, 3], ylimits=[-3, 3], numticks=101):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z)
    ax.set_yticks([])
    ax.set_xticks([])


def plot_2d_fun(f):
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plot_isocontours(ax, f)
    plt.plot([3, -3], [3, -3], 'b-')
    plt.show(block=True)
    plt.draw()


if __name__ == '__main__':

    def prior_log_density(zs):
        za, zb = zs[:, 0], zs[:, 1]
        return #TODO

    fake_games = np.array([np.repeat([1, 0], 10), np.repeat([0, 1], 10)]).T
    def log_density(zs):
        return #TODO


    def objective(params, t):
        return #TODO




    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    def callback(params, t, g):
        print("Iteration {} lower bound {}".format(t, -objective(params, t)))

        plt.cla()
        target_distribution = #TODO
        plot_isocontours(ax, target_distribution)

        mean, log_std = unpack_params(params)
        variational_contour = #TODO
        plot_isocontours(ax, variational_contour)
        plt.draw()
        plt.pause(1.0/30.0)

    print("Optimizing variational parameters...")
    init_mean    = -1 * np.ones(2)
    init_log_std = -2 * np.ones(2)
    init_var_params = np.concatenate([init_mean, init_log_std])

    variational_params = #TODO: update parameters with lr-sized step in descending gradient direction.
