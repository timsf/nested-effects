import pytest

import numpy as np
from scipy.stats import wishart

import nfx.mlm.student


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


def sample_data(bet0, t, lam, df_eta, ome):

    x = np.hstack([np.ones((t, 1)), ome.standard_normal((t, bet0.shape[1] - 1))])
    eta = ome.gamma(df_eta / 2, 2 / df_eta, (bet0.shape[0], t))
    y1 = ome.normal(bet0 @ x.T, 1 / np.sqrt(eta * lam))
    return y1, x, eta


def sample_fixture(j, l, t, lam, df_tau, df_eta, ome):

    i = sample_tree(j, ome)
    bet, tau = sample_params(i, l, df_tau, ome)
    y1, x, eta = sample_data(bet[0], t, lam, df_eta, ome)
    return (y1, x, i), (bet, tau), (lam, eta)


def test_cauchy(j=np.array([1000, 100, 10]), l=3, t=100, lam=1, n_samples=int(1e3), seed=0):

    ome = np.random.default_rng(seed)
    data, params, nuisance = sample_fixture(j, l, t, lam, 2 * l, 1, ome)
    sampler = nfx.mlm.student.sample_posterior(*data, 1, ome=ome)
    samples = [next(sampler) for _ in range(n_samples)]
