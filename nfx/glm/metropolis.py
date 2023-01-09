from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt


FloatArr = npt.NDArray[np.float_]


def sample_marginal_1o(
    x: FloatArr,
    mu: FloatArr,
    tau: FloatArr,
    delt: FloatArr,
    f_log_f: Callable[[FloatArr], Tuple[FloatArr, FloatArr, FloatArr]],
    ome: np.random.Generator,
) -> Tuple[FloatArr, FloatArr]:

    l, u = np.linalg.eigh(tau)

    l_b = l[np.newaxis] + 1 / delt[:, np.newaxis]
    l_a = 1 / l_b
    l_cov = l_a + np.square(l_a) / delt[:, np.newaxis]
    l_prec = 1 / l_cov

    x_log_f, dx_log_f, _ = f_log_f(x)
    x_log_p = x_log_f + eval_norm_prec(x, mu, u, l[np.newaxis])
    mean_x = (((x / delt[:, np.newaxis] + mu @ tau + dx_log_f) @ u) * l_a) @ u.T
    y = sample_norm_cov(mean_x, u, l_cov, ome)

    y_log_f, dy_log_f, _ = f_log_f(y)
    y_log_p = y_log_f + eval_norm_prec(y, mu, u, l[np.newaxis])
    mean_y = (((y / delt[:, np.newaxis] + mu @ tau + dy_log_f) @ u) * l_a) @ u.T

    log_post_odds = y_log_p - x_log_p
    log_prop_odds = eval_norm_prec(y, mean_x, u, l_prec) - eval_norm_prec(x, mean_y, u, l_prec)
    log_acc_odds = log_post_odds - log_prop_odds
    acc_prob = np.exp([min(0, lp) for lp in np.where(np.isnan(log_acc_odds), -np.inf, log_acc_odds)])

    return np.where(ome.uniform(size=x.shape[0]) < acc_prob, y.T, x.T).T, acc_prob


def sample_marginal_2o(
    x: FloatArr,
    mu: FloatArr,
    tau: FloatArr,
    delt: FloatArr,
    f_log_f: Callable[[FloatArr], Tuple[FloatArr, FloatArr, FloatArr]],
    ome: np.random.Generator,
) -> Tuple[FloatArr, FloatArr]:

    l, u = np.linalg.eigh(tau)

    x_log_f, dx_log_f, d2x_log_f = f_log_f(x)
    x_log_p = x_log_f + eval_norm_prec(x, mu, u, l[np.newaxis])
    y, log_prop_forw = zip(*[
        propose_2o(x_, None, mu_, tau, dx_log_f_, d2x_log_f_, delt_, ome)
        for x_, mu_, dx_log_f_, d2x_log_f_, delt_ in zip(x, mu, dx_log_f, d2x_log_f, delt)])
    y, log_prop_forw = np.array(y), np.array(log_prop_forw)

    y_log_f, dy_log_f, d2y_log_f = f_log_f(y)
    y_log_p = y_log_f + eval_norm_prec(y, mu, u, l[np.newaxis])
    _, log_prop_back = zip(*[
        propose_2o(y_, x_, mu_, tau, dy_log_f_, d2y_log_f_, delt_, ome)
        for y_, x_, mu_, dy_log_f_, d2y_log_f_, delt_ in zip(y, x, mu, dy_log_f, d2y_log_f, delt)])
    log_prop_back = np.array(log_prop_back)

    log_post_odds = y_log_p - x_log_p
    log_prop_odds = log_prop_back - log_prop_forw
    acc_prob = np.exp([min(0, lp) for lp in np.where(~np.isnan(log_post_odds), log_post_odds, -np.inf) + np.where(~np.isnan(log_prop_odds), log_prop_odds, -np.inf)])

    return np.where(ome.uniform(size=x.shape[0]) < acc_prob, y.T, x.T).T, acc_prob


def propose_2o(
    x: FloatArr,
    y: Optional[FloatArr],
    mu: FloatArr,
    tau: FloatArr,
    dx_log_f: FloatArr,
    d2x_log_f: FloatArr,
    delt: float,
    ome: np.random.Generator,
) -> Tuple[FloatArr, float]:

    tau_x = tau - d2x_log_f
    try:
        l_tau_x, u_x = np.linalg.eigh(tau_x)
    except np.linalg.LinAlgError:
        return x, np.nan
    l_b_x = l_tau_x + 1 / delt
    l_a_x = 1 / l_b_x
    l_cov_x = l_a_x + np.square(l_a_x) / delt
    l_prec_x = 1 / l_cov_x
    mean_x = (((x / delt - x @ d2x_log_f + mu @ tau + dx_log_f) @ u_x) * l_a_x) @ u_x.T
    # mean_x = ((u_tau_x / l_tau_x) @ u_tau_x.T) @ (dx_log_f + tau @ mu - d2x_log_f @ x)
    if y is None:
        y_ = sample_norm_cov(mean_x[np.newaxis], u_x, l_cov_x[np.newaxis], ome)[0]
    else:
        y_ = y
    log_prop_forw = eval_norm_prec(y_[np.newaxis], mean_x, u_x, l_prec_x[np.newaxis])[0]
    return y_, log_prop_forw


def eval_norm_prec(
    x: FloatArr,
    mu: FloatArr,
    u: FloatArr,
    l_tau: FloatArr,
) -> FloatArr:

    mah = np.sum(np.square(((x - mu) @ u) * np.sqrt(l_tau)), 1)
    return (np.sum(np.log(l_tau), 1) - mah - x.shape[1] * np.log(2 * np.pi)) / 2


def sample_norm_cov(
    mu: FloatArr,
    u: FloatArr,
    l_sig: FloatArr,
    ome: np.random.Generator,
) -> FloatArr:

    z = ome.standard_normal(mu.shape)
    return mu + (z * np.sqrt(l_sig)) @ u.T


class LatentGaussSampler(object):

    def __init__(self, n: FloatArr, opt_prob: float = .5):

        self.n = n
        self.emp_prob = [np.ones(len(n))]
        self.step = [-np.log(n)]
        self.opt_prob = opt_prob

    def sample(
        self,
        x_nil: FloatArr,
        mu: FloatArr,
        tau: FloatArr,
        f_log_f: Callable[[FloatArr], Tuple[FloatArr, FloatArr, FloatArr]],
        ome: np.random.Generator,
    ) -> FloatArr:

        x_prime, emp_prob = sample_marginal_2o(x_nil, mu, tau, np.exp(self.step[-1]), f_log_f, ome)
        self.emp_prob.append(emp_prob)
        self.step.append(self.step[-1] + (emp_prob - self.opt_prob) / np.sqrt(len(self.emp_prob)))
        return x_prime
