from typing import Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
import polyagamma as pg

import nfx.lm.gibbs
import nfx.mlm.process
import nfx.bprop.dense
import nfx.sla.dense


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]


def sample_posterior(
    y: FloatArr,
    n: FloatArr,
    x: FloatArr,
    ik: List[IntArr],
    mu0: FloatArr = None,
    tau0: FloatArr = None,
    prior_n_tau: FloatArr = None,
    prior_est_tau: List[FloatArr] = None,
    init: Tuple[List[FloatArr], List[FloatArr], FloatArr] = None,
    bprop: bool = False,
    ome: np.random.Generator = np.random.default_rng(),
) -> Iterator[Tuple[List[FloatArr], List[FloatArr], FloatArr]]:

    if mu0 is None:
        mu0 = np.zeros(x.shape[1])
    if tau0 is None:
        tau0 = np.identity(x.shape[1])
    if prior_n_tau is None:
        prior_n_tau = np.repeat(x.shape[1], len(ik) + 1)
    if prior_est_tau is None:
        prior_est_tau = (len(ik) + 1) * [np.identity(x.shape[1])]

    if init is None:
        bet = [np.zeros((len(ik_), x.shape[1])) for ik_ in ik] \
            + [np.zeros((max(ik[-1]) + 1, x.shape[1]))] + [np.zeros((1, x.shape[1]))]
        tau = prior_est_tau
        eta = np.ones_like(y)
    else:
        bet, tau, eta = init

    iik = nfx.mlm.process.reverse_edges(ik)

    while True:
        if bprop:
            ysuff = nfx.mlm.process.process_bprop(y - n / 2, x, eta)
            bet = nfx.bprop.dense.sample_nested_lm(ysuff, ik, iik, mu0, tau0, tau, np.ones(y.shape[0]), ome)
        else:
            ysuff = nfx.mlm.process.process_sla(y - n / 2, x, eta, ik, iik)
            bet = nfx.sla.dense.sample_nested_lm(ysuff, ik, mu0, tau0, tau, np.ones(y.shape[0]), ome)
        if not np.all(np.isinf(prior_n_tau)):
            tau = nfx.lm.gibbs.update_scale(ik, bet, prior_n_tau, prior_est_tau, ome)
        eta = update_latent(n, x, bet[0], True, ome)
        yield bet, tau, eta


def update_latent(
    n: FloatArr,
    x: FloatArr,
    bet0: FloatArr,
    exact: bool,
    ome: np.random.Generator,
) -> FloatArr:

    eta = bet0 @ x.T
    if exact:
        return pg.random_polyagamma(n, eta, method='gamma', random_state=ome)
    else:
        return pg.random_polyagamma(n, eta, random_state=ome)
