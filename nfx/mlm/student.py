from typing import Iterator

import numpy as np
import numpy.typing as npt

from nfx.mlm import gibbs


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]


def sample_posterior(
    y: FloatArr, 
    x: FloatArr, 
    ik: list[IntArr], 
    nu: float = 1, 
    mu0: FloatArr = None, 
    tau0: FloatArr = None,
    prior_n_tau: FloatArr = None, 
    prior_est_tau: list[FloatArr] = None,
    prior_n_lam: float = 1, 
    prior_est_lam: float = 1,
    init: gibbs.ParamSpace = None, 
    bprop: bool = False, 
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[gibbs.ParamSpace]:

    def bound_sample_latent(sq_eps_: FloatArr, ome_: np.random.Generator) -> FloatArr:
        return sample_latent(sq_eps_, nu, ome_)

    return gibbs.sample_posterior(y, x, ik, bound_sample_latent, mu0, tau0, 
                                  prior_n_tau, prior_est_tau, prior_n_lam, prior_est_lam, init, bprop, ome)


def sample_latent(sq_eps: FloatArr, nu: float, ome: np.random.Generator) -> FloatArr:

    post_n = nu + 1
    post_est = (nu + sq_eps) / post_n
    return ome.gamma(post_n / 2, 2 / (post_n * post_est))
