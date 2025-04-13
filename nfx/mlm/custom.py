from typing import Iterator

import numpy as np
import numpy.typing as npt

import nfx.mlm.gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(
    y: FloatArr, 
    x: FloatArr, 
    ik: list[IntArr],
    mu0: FloatArr = None, 
    tau0: FloatArr = None,
    prior_n_tau: FloatArr = None, 
    prior_est_tau: list[FloatArr] = None,
    prior_n_lam: float = 1, 
    prior_est_lam: float = 1, 
    prior_n_eta: FloatArr = None, 
    init: tuple[list[FloatArr], list[FloatArr], float, FloatArr] = None,
    bprop: bool = False, ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[tuple[list[FloatArr], list[FloatArr], float, FloatArr]]:

    def bound_sample_latent(sq_eps_: FloatArr, ome_: np.random.Generator) -> FloatArr:
        return sample_latent(sq_eps_, prior_n_eta, ome_)

    if prior_n_eta is None:
        prior_n_eta = np.tile(np.inf, y.shape)

    return nfx.mlm.gibbs.sample_posterior(y, x, ik, bound_sample_latent, mu0, tau0, 
                                          prior_n_tau, prior_est_tau, prior_n_lam, prior_est_lam, init, bprop, ome)


def sample_latent(sq_eps: FloatArr, prior_n: FloatArr, ome: np.random.Generator) -> FloatArr:

    post_n = prior_n + 1
    post_est = (prior_n + sq_eps) / np.where(np.isinf(post_n), 1, post_n)
    return np.where(np.isinf(prior_n), 1, ome.gamma(post_n / 2, 2 / (post_n * post_est)))
