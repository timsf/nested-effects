from typing import Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
import scipy.stats

import nfx.lm.process
import nfx.bprop.dense
import nfx.sla.dense
import nfx.misc.linalg


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]


def sample_posterior(y: FloatArr, x: FloatArr, ik: List[IntArr], 
                     mu0: FloatArr = None, tau0: FloatArr = None,
                     prior_n_tau: FloatArr = None, prior_est_tau: List[FloatArr] = None,
                     prior_n_lam: FloatArr = None, prior_est_lam: FloatArr = None,
                     init: Tuple[List[FloatArr], List[FloatArr], FloatArr] = None, bprop: bool = False, 
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[FloatArr], List[FloatArr], FloatArr]]:

    if mu0 is None:
        mu0 = np.zeros(x.shape[1])
    if tau0 is None:
        tau0 = np.zeros((x.shape[1], x.shape[1]))
    if prior_n_tau is None:
        prior_n_tau = np.repeat(x.shape[1], len(ik) + 1)
    if prior_est_tau is None:
        prior_est_tau = (len(ik) + 1) * [np.identity(x.shape[1])]
    if prior_n_lam is None:
        prior_n_lam = np.ones(y.shape[0])
    if prior_est_lam is None:
        prior_est_lam = np.ones(y.shape[0])

    if init is None:
        bet = [np.zeros((len(ik_), x.shape[1])) for ik_ in ik] \
            + [np.zeros((max(ik[-1]) + 1, x.shape[1]))] + [np.zeros((1, x.shape[1]))]
        tau = prior_est_tau
        lam = prior_est_lam
    else:
        bet, tau, lam = init

    iik = nfx.lm.process.reverse_edges(ik)
    dim_y, cyy, cxx, cxy = nfx.lm.process.eval_suff_stat(y, x)
    ysuff_bprop = nfx.lm.process.process_bprop(dim_y, cyy, cxx, cxy)
    ysuff_sla = nfx.lm.process.process_sla(dim_y, cxx, cxy, ik, iik)

    while True:
        if bprop:
            bet = nfx.bprop.dense.sample_nested_lm(ysuff_bprop, ik, iik, mu0, tau0, tau, lam, ome)
        else:
            bet = nfx.sla.dense.sample_nested_lm(ysuff_sla, ik, mu0, tau0, tau, lam, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = update_scale(ik, bet, prior_n_tau, prior_est_tau, ome)
        if not np.all(np.isinf(prior_n_lam)):
            lam = update_resid(dim_y, cyy, cxx, cxy, bet[0], prior_n_lam, prior_est_lam, ome)
        yield bet, tau, lam


def update_scale(ik: List[IntArr], bet: List[FloatArr], prior_n: FloatArr, prior_est: List[FloatArr], 
                 ome: np.random.Generator) -> List[FloatArr]:

    gam = [bet1 - bet0[ik_] for bet1, bet0, ik_ in zip(bet, bet[1:], ik + [np.int_(np.zeros(len(bet[-2])))])]
    post_n = prior_n + np.int_([len(ik_) for ik_ in ik] + [max(ik[-1]) + 1])
    post_est = [prior_est_
                    if np.isinf(prior_n_)
                    else post_n_ * nfx.misc.linalg.swm_update(prior_est_ / prior_n_, gam_, gam_)
                for prior_n_, prior_est_, post_n_, gam_ in zip(prior_n, prior_est, post_n, gam)]
    tau = [scipy.stats.wishart.rvs(post_n_, post_est_ / post_n_, random_state=ome) 
           for post_n_, post_est_ in zip(post_n, post_est)]
    return tau


def update_resid(dim_y: int, cyy: FloatArr, cxx: FloatArr, cxy: FloatArr, bet: FloatArr, 
                 prior_n: FloatArr, prior_est: FloatArr, ome: np.random.Generator) -> FloatArr:

    ssq_resid = cyy + np.sum((bet @ cxx) * bet, 1) - 2 * np.sum(bet * cxy, 1)
    post_n = prior_n + dim_y
    post_est = (prior_n * prior_est + ssq_resid) / np.where(np.isinf(post_n), 1, post_n)
    return np.where(np.isinf(prior_n), prior_est, ome.gamma(post_n / 2, 2 / (post_n * post_est)))
