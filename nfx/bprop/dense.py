from typing import List, NamedTuple

import numpy as np
import numpy.typing as npt

import scipy.linalg


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]


class F2V(NamedTuple):
    p: FloatArr
    u: FloatArr


class V2F(NamedTuple):
    p: FloatArr
    u: FloatArr
    cf_cov: FloatArr


class LmSuffStat(NamedTuple):
    cyy: float
    cxx: FloatArr
    cxy: FloatArr
    dim_y: int
    

def sample_nested_lm(y: List[LmSuffStat], ik: List[IntArr], iik: List[List[IntArr]], 
                     mu0: FloatArr, tau0: FloatArr, 
                     tau: List[FloatArr], lam: FloatArr, ome: np.random.Generator) -> List[FloatArr]:

    f2v0 = [prop_leaf_f2v_lm(y_, lam_) for y_, lam_ in zip(y, lam)]
    v2f = prop_forw(f2v0, iik, tau0, tau)
    bet = sample_backw(v2f, ik, mu0, tau0, tau, ome)
    return bet


def prop_forw(f2v0: List[F2V], iik: List[List[IntArr]], tau0: FloatArr, tau: List[FloatArr]
              ) -> List[List[V2F]]:

    f2v = [f2v0]
    v2f = []
    for tau_, iik_ in zip(tau, [[np.int_([i]) for i in range(len(f2v0))]] + iik):
        v2f.append(prop_v2f(f2v[-1], iik_, tau_))
        f2v.append([prop_f2v(msg, tau_) for msg in v2f[-1]])
    v2f.append(prop_v2f(f2v[-1], [np.arange(len(iik[-1]))], tau0))
    return v2f


def prop_leaf_f2v_lm(y: LmSuffStat, lam: float) -> F2V:

    p = lam * y.cxx
    u = lam * y.cxy
    return F2V(p, u)


def prop_f2v(v2f: V2F, tau: FloatArr) -> F2V:

    coefs = tau @ v2f.cf_cov.T @ v2f.cf_cov
    p = coefs @ v2f.p
    u = coefs @ v2f.u
    return F2V(p, u)


def prop_v2f(f2v: List[F2V], iik: List[IntArr], tau: FloatArr) -> List[V2F]:

    def prop_v2f_inner(msg: List[F2V]) -> V2F:

        p, u = (sum(a) for a in zip(*msg))
        cf_cov = scipy.linalg.solve_triangular(np.linalg.cholesky(tau + p), np.identity(len(u)), 
                                               lower=True, check_finite=False)
        return V2F(p, u, cf_cov)

    return [prop_v2f_inner([f2v[i] for i in iik_]) for iik_ in iik]


def sample_backw(v2f: List[List[V2F]], ik: List[IntArr], mu0: FloatArr, tau0: FloatArr, tau: List[FloatArr], 
                 ome: np.random.Generator) -> List[FloatArr]:

    bet = [sample_node(v2f[-1][0], mu0, tau0, ome)[np.newaxis]]
    for v2f_, tau_, ik_ in list(zip(v2f[:-1], tau, ik + [np.int_(np.zeros(len(v2f[-2])))]))[::-1]:
        bet.append(np.vstack([sample_node(msg, mu, tau_, ome) for msg, mu in zip(v2f_, bet[-1][ik_])]))
    return bet[::-1]


def sample_node(v2f: V2F, mu: FloatArr, tau: FloatArr, ome: np.random.Generator) -> FloatArr:

    z = ome.standard_normal(len(mu))
    return v2f.cf_cov.T @ (v2f.cf_cov @ (v2f.u + tau @ mu) + z)


# def sample_nested(y: np.ndarray, ik: List[np.ndarray], iik: List[List[np.ndarray]], lam: np.ndarray, 
#                   mu0: np.ndarray, tau0: np.ndarray, tau: List[np.ndarray], ome: np.random.Generator
#                   ) -> List[np.ndarray]:

#     f2v0 = [prop_leaf_f2v_lm(y_, lam) for y_ in y]
#     v2f = prop_forw(f2v0, iik, tau0, tau)
#     bet = sample_backw(v2f, ik, mu0, tau0, tau, ome)
#     return bet


# def prop_leaf_f2v(y: np.ndarray, lam: np.ndarray) -> F2V:

#     p = lam
#     u = lam @ y
#     return F2V(p, u)
