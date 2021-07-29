import pytest

import numpy as np

import nfx.lm.gibbs
from scipy.stats import wishart


def sample_tree(j, ome):

    i = [np.zeros(j[-1], dtype=np.int64)]
    for j0, j1 in zip(j[::-1], j[::-1][1:]):
        i.append(ome.permutation(np.repeat(np.arange(j0), np.ceil(j1 / j0))[:j1]))
    return i[::-1]


def sample_params(i, l, df_tau, ome):

    tau = [wishart.rvs(df_tau, np.identity(l) / df_tau, random_state=ome) for _ in range(len(i))]
    bet = sample_coefs(i, tau, ome)
    return bet, tau


def sample_coefs(i, tau, ome):

    bet = [ome.standard_normal((1, len(tau[0])))]
    for tau_, i_ in zip(tau[::-1], i[::-1]):
        bet.append(bet[-1][i_] + np.linalg.solve(tau_, ome.standard_normal((len(tau_), len(i_)))).T)
    return bet[::-1]


def sample_data(bet0, t, lam, ome):

    x = np.hstack([np.ones((t, 1)), ome.standard_normal((t, bet0.shape[1] - 1))])
    y1 = ome.normal(bet0 @ x.T, 1 / np.sqrt(lam))
    return y1, x


def sample_fixture(j, l, t, lam, ome):

    i = sample_tree(j, ome)
    bet, tau = sample_params(i, l, 2 * l, ome)
    y1, x = sample_data(bet[0], t, lam, ome)
    return (y1, x, i), (bet, tau), (lam,)


def test_lm(j=np.array([1000, 100, 10]), l=3, t=100, lam=1, n_samples=int(1e3), seed=0):

    ome = np.random.default_rng(seed)
    data, params, hyper = sample_fixture(j, l, t, lam, ome)
    sampler = nfx.lm.gibbs.sample_posterior(*data, ome=ome)
    samples = [next(sampler) for _ in range(n_samples)]
