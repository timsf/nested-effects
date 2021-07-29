from typing import Iterator, List, Tuple

import numpy as np

from nfx.glm import gibbs


def sample_posterior(y1: np.ndarray, y2: np.ndarray, n: np.ndarray, x: np.ndarray, ik: List[np.ndarray],
                     mu0: np.ndarray = None, tau0: np.ndarray = None,
                     prior_n_tau: np.ndarray = None, prior_est_tau: List[np.ndarray] = None, 
                     prior_n_phi: float = 1, prior_est_phi: float = 1,
                     init: Tuple[List[np.ndarray], List[np.ndarray], float] = None, bprop: bool = False,
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray], float]]:

    return gibbs.sample_disp_posterior(y1, y2, n, x, ik, eval_part, eval_base, mu0, tau0,
                                       prior_n_tau, prior_est_tau, prior_n_phi, prior_est_phi,
                                       init, bprop, ome)


def eval_part(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    return np.square(eta) / 2, eta, np.ones_like(eta)


def eval_base(y1: np.ndarray, y2: np.ndarray, n: np.ndarray, phi: float) -> Tuple[float, float, float]:

    log_g = - np.sum(n) * np.log(2 * np.pi * phi) / 2 - np.sum(y2) / (2 * phi)
    d_log_g = - np.sum(n) / (2 * phi) + np.sum(y2) / (2 * phi ** 2)
    d2_log_g = np.sum(n) / (2 * phi ** 2) - np.sum(y2) / phi ** 3
    return log_g, d_log_g, d2_log_g
    