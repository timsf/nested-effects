from typing import Callable, Tuple

import numpy as np


def sample(x: np.ndarray, mu: np.ndarray, tau: np.ndarray, sig: np.ndarray,
           f_log_p: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]], ome: np.random.Generator
           ) -> Tuple[np.ndarray, np.ndarray]:

    x_log_p, mean_x, prec_x = ascend(x, mu, tau, sig, f_log_p)
    y = ome.normal(mean_x, 1 / np.sqrt(prec_x))
    y_log_p, mean_y, prec_y = ascend(y, mu, tau, sig, f_log_p)
    return accept_reject(x, y, x_log_p, y_log_p, mean_x, mean_y, prec_x, prec_y, mu, tau, ome)


def ascend(x: np.ndarray, mu: np.ndarray, tau: np.ndarray, sig: np.ndarray,
                  f_log_p: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    x_log_p, dx_log_p, d2x_log_p = f_log_p(x)
    a = 1 / (np.ones_like(x) / sig + tau - d2x_log_p)
    x_hess = 1 / (a ** 2 / sig + a)
    x_prime = (dx_log_p + tau * mu + x * (1 / sig - d2x_log_p)) * a
    return x_log_p, x_prime, x_hess


def accept_reject(x: np.ndarray, y: np.ndarray, x_log_p: np.ndarray, y_log_p: np.ndarray,
                  mean_x: np.ndarray, mean_y: np.ndarray, prec_x: np.ndarray, prec_y: np.ndarray,
                  mu: np.ndarray, tau: np.ndarray, ome: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:

    log_lik_ratio = y_log_p - x_log_p
    log_prior_odds = eval_norm(y, mu, tau) - eval_norm(x, mu, tau)
    log_prop_odds = eval_norm(y, mean_x, prec_x) - eval_norm(x, mean_y, prec_y)
    log_prop_odds_clean = np.where(np.isnan(log_prop_odds), np.inf, log_prop_odds)
    acc_prob = np.exp([min(0, lp) for lp in log_lik_ratio + log_prior_odds - log_prop_odds_clean])
    return np.where(ome.uniform(size=len(acc_prob)) < acc_prob, y.T, x.T).T, acc_prob


def eval_norm(x: np.ndarray, mu: np.ndarray, tau: np.ndarray) -> np.ndarray:

    d = (x - mu) ** 2 * tau
    kern = -d / 2
    cons = (np.log(tau) - np.log(2 * np.pi)) / 2
    return cons + kern


def cond_norm(x: np.ndarray, mu: np.ndarray, tau: np.ndarray, update: int):

    sig = np.linalg.inv(tau)
    x_cl = np.delete(x, update, 1)
    mu_cl = np.delete(mu, update, 1)
    tau_cl_cl = np.linalg.inv(np.delete(np.delete(sig, update, 0), update, 1))
    coefs = np.delete(sig[update], update) @ tau_cl_cl
    return mu[:, update] - coefs @ (mu_cl - x_cl).T, tau[update, update]


class LatentGaussSampler(object):

    def __init__(self, j: int, l: int, opt_prob: float = .5):

        self.emp_prob = [[np.ones(j)] for _ in range(l)]
        self.step = [[np.zeros(j)] for _ in range(l)]
        self.opt_prob = opt_prob

    def sample(self, x_nil: np.ndarray, mu: np.ndarray, tau: np.ndarray,
               f_log_p: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]], ome: np.random.Generator
               ) -> np.ndarray:

        def f_log_p_cond(x_prime_cond: np.ndarray):
            x_prime_ = np.copy(x_prime)
            x_prime_[:, l] = x_prime_cond
            log_p, d_log_p, d2_log_p = f_log_p(x_prime_)
            return log_p, d_log_p[:, l], d2_log_p[:, l]

        x_prime = np.copy(x_nil)
        for l in range(x_nil.shape[1]):
            mu_cond, tau_cond = cond_norm(x_nil, mu, tau, l)
            x_prime_cond, emp_prob = sample(x_nil[:, l], mu_cond, np.repeat(tau_cond, len(mu_cond)), 
                                            np.exp(self.step[l][-1]), f_log_p_cond, ome)
            x_prime[:, l] = x_prime_cond
            self.emp_prob[l].append(emp_prob)
            self.step[l].append(self.step[l][-1] + (emp_prob - self.opt_prob) / np.sqrt(len(self.emp_prob[l])))
        return x_prime
