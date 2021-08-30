from typing import Iterator, List, Tuple

import numpy as np

from nfx.glm import gibbs


def sample_posterior(y1: np.ndarray, n: np.ndarray, x: np.ndarray, ik: List[np.ndarray],
                     mu0: np.ndarray = None, tau0: np.ndarray = None,
                     prior_n_tau: np.ndarray = None, prior_est_tau: List[np.ndarray] = None,
                     init: Tuple[List[np.ndarray], List[np.ndarray]] = None, bprop: bool = False,
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray]]]:

    return gibbs.sample_posterior(y1, n, x, ik, eval_part, mu0, tau0, prior_n_tau, prior_est_tau, init, bprop, ome)


def eval_part(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    fitted = np.exp(eta)
    fitted = np.where(np.isinf(fitted), np.nan, fitted)
    return fitted, fitted, fitted
