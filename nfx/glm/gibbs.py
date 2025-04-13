from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import root_scalar

import nfx.lm.gibbs
import nfx.glm.metropolis
import nfx.glm.process
import nfx.bprop.dense
import nfx.sla.dense


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float64]
PartFunc = Callable[[FloatArr], Tuple[FloatArr, FloatArr, FloatArr]]
BaseFunc = Callable[[FloatArr, FloatArr, FloatArr, float], Tuple[float, float, float]]


def sample_disp_posterior(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    x: FloatArr,
    ik: List[IntArr],
    eval_part: PartFunc,
    eval_base: BaseFunc,
    mu0: Optional[FloatArr],
    tau0: Optional[FloatArr],
    prior_n_tau: Optional[FloatArr],
    prior_est_tau: Optional[List[FloatArr]],
    prior_n_phi: Optional[float],
    prior_est_phi: Optional[float],
    init: Optional[Tuple[List[FloatArr], List[FloatArr], float]],
    bprop: bool,
    ome: np.random.Generator,
) -> Iterator[Tuple[List[FloatArr], List[FloatArr], float]]:

    if mu0 is None:
        mu0 = np.zeros(x.shape[1])
    if tau0 is None:
        tau0 = np.identity(x.shape[1])
    if prior_n_tau is None:
        prior_n_tau = np.repeat(x.shape[1], len(ik) + 1)
    if prior_est_tau is None:
        prior_est_tau = (len(ik) + 1) * [np.identity(x.shape[1])]
    if prior_n_phi is None:
        prior_n_phi = 1
    if prior_est_phi is None:
        prior_est_phi = 1

    if init is None:
        bet0 = np.zeros((len(ik[0]), x.shape[1]))
        bet = [np.zeros((len(ik_), x.shape[1])) for ik_ in ik[1:]] \
            + [np.zeros((max(ik[-1]) + 1, x.shape[1]))] + \
            [np.zeros((1, x.shape[1]))]
        tau = prior_est_tau
        phi = prior_est_phi
    else:
        bet0 = init[0][0]
        bet = init[0][1:]
        tau, phi = init[1:]

    iik = nfx.glm.process.reverse_edges(ik)
    sampler = nfx.glm.metropolis.LatentGaussSampler(np.sum(n, 1))

    while True:
        bet0 = update_leaves(y1, n, x, ik, bet0, bet, tau, phi, eval_part, sampler, ome)
        if bprop:
            asuff_bprop = nfx.glm.process.process_bprop(bet0, tau[0], ik, iik)
            bet = nfx.bprop.dense.sample_nested_lm(
                asuff_bprop, ik[1:], iik[1:], mu0, tau0, tau[1:], np.ones(len(iik[0])), ome)
        else:
            asuff_sla = nfx.glm.process.process_sla(bet0, tau[0], ik, iik)
            bet = nfx.sla.dense.sample_nested_lm(
                asuff_sla, ik[1:], mu0, tau0, tau[1:], np.ones(len(iik[0])), ome)
        # if not np.all(np.isinf(prior_n_tau)):
        #     tau = nfx.lm.gibbs.update_scale(ik, [bet0] + bet, prior_n_tau, prior_est_tau, ome)
        if not np.isinf(prior_n_phi):
            phi = update_dispersion(y1, y2, n, x, bet0, phi, eval_part, eval_base, prior_n_phi, prior_est_phi, ome)
        yield [bet0] + bet, tau, phi


def sample_posterior(
    y1: FloatArr,
    n: FloatArr,
    x: FloatArr,
    ik: List[IntArr],
    eval_part: PartFunc,
    mu0: Optional[FloatArr],
    tau0: Optional[FloatArr],
    prior_n_tau: Optional[FloatArr],
    prior_est_tau: Optional[List[FloatArr]],
    init: Optional[Tuple[List[FloatArr], List[FloatArr]]],
    bprop: bool,
    ome: np.random.Generator,
) -> Iterator[Tuple[List[FloatArr], List[FloatArr]]]:

    def eval_base(_, __, ___, ____): return (0, 0, 0)
    return (the[:-1] for the in
            sample_disp_posterior(y1, np.empty_like(y1), n, x, ik, eval_part, eval_base, prior_n_tau, mu0, tau0,
                                  prior_est_tau, np.inf, 1, init + (1,) if init is not None else init, bprop, ome))


def update_leaves(
    y1: FloatArr,
    n: FloatArr,
    x: FloatArr,
    ik: List[IntArr],
    bet0: FloatArr,
    bet: List[FloatArr],
    tau: List[FloatArr],
    phi: float,
    eval_part: PartFunc,
    sampler: nfx.glm.metropolis.LatentGaussSampler,
    ome: np.random.Generator,
) -> FloatArr:

    def eval_log_f(b0: FloatArr) -> Tuple[FloatArr, FloatArr, FloatArr]:
        log_p, d_log_p, d2_log_p = eval_loglik(y1, n, x, b0, eval_part)
        return log_p / phi, d_log_p / phi, d2_log_p / phi

    new_bet0 = sampler.sample(bet0, bet[0][ik[0]], tau[0], eval_log_f, ome)
    return new_bet0


def update_dispersion(
    y1: FloatArr,
    y2: FloatArr,
    n: FloatArr,
    x: FloatArr,
    bet0: FloatArr,
    phi: float,
    eval_part: PartFunc,
    eval_base: BaseFunc,
    prior_n: float,
    prior_est: float,
    ome: np.random.Generator,
) -> float:

    def eval_log_p(phi_: float, log_v: float) -> Tuple[float, float, float]:
        log_g, d_log_g, d2_log_g = eval_base(y1, y2, n, phi_)
        log_prior, d_log_prior, d2_log_prior = eval_logprior_phi(phi_, prior_n, prior_est)
        log_p = log_prior + log_g + log_p_nil / phi_ - log_v
        d_log_p = d_log_prior + d_log_g - log_p_nil / phi_ ** 2
        d2_log_p = d2_log_prior + d2_log_g + 2 * log_p_nil / phi_ ** 3
        return log_p, d_log_p, d2_log_p

    def brace(right: bool) -> float:
        sgn = 1 if right else -1
        width = 1
        while True:
            edge = float(phi * 2 ** (sgn * width))
            log_p, d_log_p, _ = eval_log_p(edge, log_u)
            if log_p < 0 and sgn * d_log_p < 0:
                return edge
            width += 1

    log_p_nil = np.sum(eval_loglik(y1, n, x, bet0, eval_part)[0])
    log_u = eval_log_p(phi, 0)[0] - ome.exponential()
    lb = root_scalar(eval_log_p, (log_u,), bracket=(brace(False), phi), fprime=True, fprime2=True).root
    ub = root_scalar(eval_log_p, (log_u,), bracket=(phi, brace(True)), fprime=True, fprime2=True).root
    return ome.uniform(lb, ub)


def eval_loglik(
    y1: FloatArr,
    n: FloatArr,
    x: FloatArr,
    bet0: FloatArr,
    eval_part: PartFunc,
) -> Tuple[FloatArr, FloatArr, FloatArr]:

    eta = bet0 @ x.T
    part, d_part, d2_part = eval_part(eta)
    log_f = np.sum(y1 * eta - n * part, 1)
    d_log_f = (y1 - n * d_part) @ x
    # d2_log_f = -(n * d2_part) @ np.square(x)
    sqrt_d2_log_f = np.sqrt(n * d2_part)[:, :, np.newaxis] * x[np.newaxis]
    d2_log_f = -np.array([x_.T @ x_ for x_ in sqrt_d2_log_f])
    return log_f, d_log_f, d2_log_f


def eval_logprior_phi(phi: float, prior_n: float, prior_est: float) -> Tuple[float, float, float]:

    return -(prior_n / 2 + 1) * np.log(phi) - prior_n * prior_est / (2 * phi), \
           -(prior_n / 2 + 1) / phi + prior_n * prior_est / (2 * phi ** 2), \
           (prior_n / 2 + 1) / phi ** 2 - prior_n * prior_est / phi ** 3
