from typing import Iterator, List, Tuple

import numpy as np

import nfx.mlm.gibbs


def sample_posterior(y: np.ndarray, x: np.ndarray, ik: List[np.ndarray],
                     mu0: np.ndarray = None, tau0: np.ndarray = None,
                     prior_n_tau: np.ndarray = None, prior_est_tau: List[np.ndarray] = None,
                     prior_n_lam: float = 1, prior_est_lam: float = 1, prior_n_eta: np.ndarray = None, 
                     init: Tuple[List[np.ndarray], List[np.ndarray], float, np.ndarray] = None,
                     bprop: bool = False, ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray], float, np.ndarray]]:

    def bound_sample_latent(sq_eps_: np.ndarray, ome_: np.random.Generator) -> np.ndarray:
        return sample_latent(sq_eps_, prior_n_eta, ome_)

    if prior_n_eta is None:
        prior_n_eta = np.tile(np.inf, y.shape)

    return nfx.mlm.gibbs.sample_posterior(y, x, ik, bound_sample_latent, mu0, tau0, 
                                          prior_n_tau, prior_est_tau, prior_n_lam, prior_est_lam, init, bprop, ome)


def sample_latent(sq_eps: np.ndarray, prior_n: np.ndarray, ome: np.random.Generator) -> np.ndarray:

    post_n = prior_n + 1
    post_est = (prior_n + sq_eps) / np.where(np.isinf(post_n), 1, post_n)
    return np.where(np.isinf(prior_n), 1, ome.gamma(post_n / 2, 2 / (post_n * post_est)))
