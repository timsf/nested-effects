from typing import NamedTuple, Optional

import numpy as np

import scipy.linalg

import nfx.misc.linalg


class F2V(NamedTuple):
    p: np.ndarray
    u: np.ndarray


class V2F(NamedTuple):
    p: np.ndarray
    u: np.ndarray
    cf_cov: np.ndarray


class LmSuffStat(NamedTuple):
    cyy: float
    cxx: np.ndarray
    cxy: np.ndarray
    dim_y: int
    

def sample_nested_lm(y: [LmSuffStat], ik: [np.ndarray], iik: [[np.ndarray]], mu0: np.ndarray, tau0: np.ndarray, 
                     tau: [np.ndarray], lam: np.ndarray, ome: np.random.Generator) -> [np.ndarray]:

    f2v0 = [prop_leaf_f2v_lm(y_, lam_) for y_, lam_ in zip(y, lam)]
    v2f = prop_forw(f2v0, iik, tau0, tau)
    bet = sample_backw(v2f, ik, mu0, tau0, tau, ome)
    return bet


def sample_nested(y: np.ndarray, ik: [np.ndarray], iik: [[np.ndarray]], lam: np.ndarray, 
                  mu0: np.ndarray, tau0: np.ndarray, tau: [np.ndarray], ome: np.random.Generator) -> [np.ndarray]:

    f2v0 = [prop_leaf_f2v(y_, iik, lam) for y_ in y]
    v2f = prop_forw(f2v0, iik, tau0, tau)
    bet = sample_backw(v2f, ik, mu0, tau0, tau, ome)
    return bet


def prop_forw(f2v0: [F2V], iik: [[np.ndarray]], tau0: np.ndarray, tau: [np.ndarray]) -> [[V2F]]:

    f2v = [f2v0]
    v2f = []
    for tau_, iik_ in zip(tau, [[np.int64([i]) for i in range(len(f2v0))]] + iik):
        v2f.append(prop_v2f(f2v[-1], iik_, tau_))
        f2v.append([prop_f2v(msg, tau_) for msg in v2f[-1]])
    v2f.append(prop_v2f(f2v[-1], [np.arange(len(iik[-1]))], tau0))
    return v2f


def prop_leaf_f2v_lm(y: LmSuffStat, lam: float) -> F2V:

    p = lam * y.cxx
    u = lam * y.cxy
    return F2V(p, u)


def prop_leaf_f2v(y: np.ndarray, lam: np.ndarray) -> F2V:

    p = lam
    u = lam @ y
    return F2V(p, u)


def prop_f2v(v2f: V2F, tau: np.ndarray) -> F2V:

    coefs = tau @ v2f.cf_cov.T @ v2f.cf_cov
    p = coefs @ v2f.p
    u = coefs @ v2f.u
    return F2V(p, u)


def prop_v2f(f2v: [F2V], iik: [np.ndarray], tau: np.ndarray) -> [V2F]:

    def prop_v2f_inner(msg: [F2V]) -> V2F:

        p, u = (sum(a) for a in zip(*msg))
        cf_cov = scipy.linalg.solve_triangular(np.linalg.cholesky(tau + p), np.identity(len(u)), lower=True, check_finite=False)
        return V2F(p, u, cf_cov)

    return [prop_v2f_inner([f2v[i] for i in iik_]) for iik_ in iik]


def sample_backw(v2f: [[V2F]], ik: [np.ndarray], mu0: np.ndarray, tau0: np.ndarray, tau: [np.ndarray], 
                 ome: np.random.Generator) -> [np.ndarray]:

    bet = [sample_node(v2f[-1][0], mu0, tau0, ome)[np.newaxis]]
    for v2f_, tau_, ik_ in list(zip(v2f[:-1], tau, ik + [np.int64(np.zeros(len(v2f[-2])))]))[::-1]:
        bet.append(np.vstack([sample_node(msg, mu, tau_, ome) for msg, mu in zip(v2f_, bet[-1][ik_])]))
    return bet[::-1]


def sample_node(v2f: V2F, mu: np.ndarray, tau: np.ndarray, ome: np.random.Generator) -> np.ndarray:

    z = ome.standard_normal(len(mu))
    return v2f.cf_cov.T @ (v2f.cf_cov @ (v2f.u + tau @ mu) + z)