import numpy as np

import nfx.bprop.dense
import nfx.sla.dense


def process_bprop(bet0: np.ndarray, tau_: np.ndarray, ik: np.ndarray, iik: [[np.ndarray]]) -> [nfx.bprop.dense.LmSuffStat]:

    eta, s1, s2 = zip(*[(len(iik_), np.mean(bet0[iik_], 0), np.sum(np.mean(bet0[iik_], 0) ** 2)) for iik_ in iik[0]])
    return [nfx.bprop.dense.LmSuffStat(s2_, n_ * tau_, n_ * (tau_ @ s1_), len(s1_)) for n_, s1_, s2_ in zip(eta, s1, s2)]


def process_sla(bet0: np.ndarray, tau_: np.ndarray, ik: np.ndarray, iik: [[np.ndarray]]) -> nfx.sla.dense.LmSuffStat:
    
    eta = np.array([len(iik_) for iik_ in iik[0]])
    y = np.array([np.mean(bet0[iik_], 0) for iik_ in iik[0]])
    n_offspring = [[len(iik__) for iik__ in iik_] for iik_ in iik[1:]] + [[len(iik[-1])]]
    n_offspring_flat = np.repeat(np.hstack(n_offspring), bet0.shape[1] ** 2)
    row_ix, col_ix = nfx.sla.dense.prepare_sparse_indices(ik[1:], bet0.shape[1])
    cxy_flat = (eta[:, np.newaxis] * (y @ tau_)).flatten()
    cxx_flat = (eta[:, np.newaxis, np.newaxis] * tau_[np.newaxis]).flatten()
    return nfx.sla.dense.LmSuffStat(row_ix, col_ix, n_offspring_flat, cxx_flat, cxy_flat)


def reverse_edges(ik: [np.ndarray]) -> [[np.ndarray]]:

    return [[np.where(ik_ == i)[0] for i in range(max(ik_) + 1)] for ik_ in ik]
