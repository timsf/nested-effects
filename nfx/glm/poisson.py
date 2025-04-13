from typing import Iterator

import numpy as np
import numpy.typing as npt

from nfx.glm import gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(
    y1: FloatArr,
    n: FloatArr,
    x: FloatArr,
    ik: list[IntArr],
    mu0: FloatArr = None,
    tau0: FloatArr = None,
    prior_n_tau: FloatArr = None,
    prior_est_tau: list[FloatArr] = None,
    init: gibbs.ParamSpace = None,
    bprop: bool = False,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.ParamSpace]:

    return gibbs.sample_posterior(y1, n, x, ik, eval_part, mu0, tau0, prior_n_tau, prior_est_tau, init, bprop, ome)


def eval_part(eta: FloatArr) -> tuple[FloatArr, FloatArr, FloatArr]:

    fitted = np.exp(eta)
    fitted = np.where(np.isinf(fitted), np.nan, fitted)
    return fitted, fitted, fitted
