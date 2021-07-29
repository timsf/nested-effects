from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

import nfx.lm.gibbs
import nfx.mlm.process
import nfx.bprop.dense
import nfx.sla.dense


def sample_posterior(y: np.ndarray, x: np.ndarray, ik: List[np.ndarray],
                     sample_latent: Callable[[np.ndarray, np.random.Generator], np.ndarray], 
                     mu0: Optional[np.ndarray], tau0: Optional[np.ndarray],
                     prior_n_tau: Optional[np.ndarray], prior_est_tau: Optional[List[np.ndarray]], 
                     prior_n_lam: Optional[float], prior_est_lam: Optional[float],
                     init: Optional[Tuple[List[np.ndarray], List[np.ndarray], float, np.ndarray]], bprop: bool,
                     ome: np.random.Generator
                     ) -> Iterator[Tuple[List[np.ndarray], List[np.ndarray], float, np.ndarray]]:

    if mu0 is None:
        mu0 = np.zeros(x.shape[1])
    if tau0 is None:
        tau0 = np.zeros((x.shape[1], x.shape[1]))
    if prior_n_tau is None:
        prior_n_tau = np.repeat(x.shape[1], len(ik) + 1)
    if prior_est_tau is None:
        prior_est_tau = (len(ik) + 1) * [np.identity(x.shape[1])]
    if prior_n_lam is None:
        prior_n_lam = 1
    if prior_est_lam is None:
        prior_est_lam = 1

    if init is None:
        bet = [np.zeros((len(ik_), x.shape[1])) for ik_ in ik] \
            + [np.zeros((max(ik[-1]) + 1, x.shape[1]))] + [np.zeros((1, x.shape[1]))]
        tau = prior_est_tau
        lam = prior_est_lam
        eta = np.ones_like(y)
    else:
        bet, tau, lam, eta = init

    iik = nfx.mlm.process.reverse_edges(ik)

    while True:
        if bprop:
            ysuff = nfx.mlm.process.process_bprop(y, x, eta)
            bet = nfx.bprop.dense.sample_nested_lm(ysuff, ik, iik, mu0, tau0, tau, np.repeat(lam, y.shape[0]), ome)
        else:
            ysuff = nfx.mlm.process.process_sla(y, x, eta, ik, iik)
            bet = nfx.sla.dense.sample_nested_lm(ysuff, ik, mu0, tau0, tau, np.repeat(lam, y.shape[0]), ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = nfx.lm.gibbs.update_scale(ik, bet, prior_n_tau, prior_est_tau, ome)
        if not np.isinf(prior_n_lam):
            lam = update_resid(y, x, bet[0], eta, prior_n_lam, prior_est_lam, ome)
        eta = update_latent(y, x, bet[0], lam, sample_latent, ome)
        yield bet, tau, lam, eta


def update_resid(y: np.ndarray, x: np.ndarray, bet: np.ndarray, eta: np.ndarray, 
                 prior_n: np.ndarray, prior_est: np.ndarray, ome: np.random.Generator) -> float:

    ssq_eps = np.sum(np.square(y - bet @ x.T) * eta)
    post_n = prior_n + np.prod(y.shape)
    post_est = prior_est if np.isinf(prior_n) else (prior_n * prior_est + ssq_eps) / post_n
    return prior_est if np.isinf(prior_n) else ome.gamma(post_n / 2, 2 / (post_n * post_est))


def update_latent(y: np.ndarray, x: np.ndarray, bet: np.ndarray, lam: float, 
                  sample_latent: Callable[[np.ndarray], np.ndarray], ome: np.random.Generator) -> float:

    return sample_latent(np.square(y - bet @ x.T) * lam, ome)
