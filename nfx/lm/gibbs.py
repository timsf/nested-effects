from typing import Iterator, List, Tuple

import numpy as np
import scipy.stats

import nfx.lm.process
import nfx.bprop.dense
import nfx.sla.dense
import nfx.misc.linalg


def sample_posterior(y: np.ndarray, x: np.ndarray, ik: List[np.ndarray], 
                     mu0: np.ndarray = None, tau0: np.ndarray = None,
                     prior_n_tau: np.ndarray = None, prior_est_tau: List[np.ndarray] = None,
                     prior_n_lam: np.ndarray = None, prior_est_lam: np.ndarray = None,
                     init: Tuple[List[np.ndarray], List[np.ndarray], np.ndarray] = None, bprop: bool = False, 
                     ome: np.random.Generator = np.random.default_rng()
                     ) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]]:

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
    if bprop:
        ysuff = nfx.lm.process.process_bprop(dim_y, cyy, cxx, cxy)
    if not bprop:
        ysuff = nfx.lm.process.process_sla(dim_y, cxx, cxy, ik, iik)

    while True:
        if bprop:
            bet = nfx.bprop.dense.sample_nested_lm(ysuff, ik, iik, mu0, tau0, tau, lam, ome)
        else:
            bet = nfx.sla.dense.sample_nested_lm(ysuff, ik, mu0, tau0, tau, lam, ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = update_scale(ik, bet, prior_n_tau, prior_est_tau, ome)
        if not np.all(np.isinf(prior_n_lam)):
            lam = update_resid(dim_y, cyy, cxx, cxy, bet[0], prior_n_lam, prior_est_lam, ome)
        yield bet, tau, lam


def update_scale(ik: List[np.ndarray], bet: List[np.ndarray], prior_n: np.ndarray, prior_est: List[np.ndarray], 
                 ome: np.random.Generator) -> List[np.ndarray]:

    gam = [bet1 - bet0[ik_] for bet1, bet0, ik_ in zip(bet, bet[1:], ik + [np.int64(np.zeros(len(bet[-2])))])]
    post_n = prior_n + np.int64([len(ik_) for ik_ in ik] + [max(ik[-1]) + 1])
    post_est = [prior_est_
                    if np.isinf(prior_n_)
                    else post_n_ * nfx.misc.linalg.swm_update(prior_est_ / prior_n_, gam_, gam_)
                for prior_n_, prior_est_, post_n_, gam_ in zip(prior_n, prior_est, post_n, gam)]
    tau = [scipy.stats.wishart.rvs(post_n_, post_est_ / post_n_, random_state=ome) 
           for post_n_, post_est_ in zip(post_n, post_est)]
    return tau


def update_resid(dim_y: int, cyy: np.ndarray, cxx: np.ndarray, cxy: np.ndarray, bet: np.ndarray, 
                 prior_n: np.ndarray, prior_est: np.ndarray, ome: np.random.Generator) -> np.ndarray:

    ssq_resid = cyy + np.sum((bet @ cxx) * bet, 1) - 2 * np.sum(bet * cxy, 1)
    post_n = prior_n + dim_y
    post_est = (prior_n * prior_est + ssq_resid) / np.where(np.isinf(post_n), 1, post_n)
    return np.where(np.isinf(prior_n), prior_est, ome.gamma(post_n / 2, 2 / (post_n * post_est)))