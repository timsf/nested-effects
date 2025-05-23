from typing import Iterator

import numpy as np
import numpy.typing as npt

from nfx.glm import gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    x: FloatArr,
    ik: list[IntArr],
    mu0: FloatArr = None,
    tau0: FloatArr = None,
    prior_n_tau: FloatArr = None,
    prior_est_tau: list[FloatArr] = None,
    prior_n_phi: float = 1,
    prior_est_phi: float = 1,
    init: gibbs.DispParamSpace = None,
    bprop: bool = False,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.DispParamSpace]:

    return gibbs.sample_disp_posterior(y1, y2, n, x, ik, eval_part, eval_base, mu0, tau0,
                                       prior_n_tau, prior_est_tau, prior_n_phi, prior_est_phi,
                                       init, bprop, ome)


def eval_part(eta: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:

    return np.square(eta) / 2, eta, np.ones_like(eta)


def eval_base(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    phi: float,
) -> tuple[float, float, float]:

    log_g = - np.sum(n) * np.log(2 * np.pi * phi) / 2 - np.sum(y2) / (2 * phi)
    d_log_g = - np.sum(n) / (2 * phi) + np.sum(y2) / (2 * phi ** 2)
    d2_log_g = np.sum(n) / (2 * phi ** 2) - np.sum(y2) / phi ** 3
    return log_g, d_log_g, d2_log_g
