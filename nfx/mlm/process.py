from typing import List

import numpy as np

import nfx.bprop.dense
import nfx.sla.dense


def process_bprop(y: np.ndarray, x: np.ndarray, eta: np.ndarray) -> List[nfx.bprop.dense.LmSuffStat]:

    ysuff = [nfx.bprop.dense.LmSuffStat(np.sum(np.square(y_)), (x.T * eta_) @ x, (y_ * eta_) @ x, len(y_)) 
             for y_, eta_ in zip(y, eta)]
    return ysuff


def process_sla(y: np.ndarray, x: np.ndarray, eta: np.ndarray, ik: List[np.ndarray], iik: List[List[np.ndarray]]
                ) -> nfx.sla.dense.LmSuffStat:
    
    n_offspring = [[len(iik__) for iik__ in iik_] for iik_ in iik] + [[len(iik[-1])]]
    n_offspring_flat = np.repeat(np.hstack(n_offspring), x.shape[1] ** 2)
    row_ix, col_ix = nfx.sla.dense.prepare_sparse_indices(ik, x.shape[1])
    cxy_flat = ((y * eta) @ x).flatten()
    cxx_flat = np.hstack([((x.T * eta_) @ x).flatten() for eta_ in eta])
    return nfx.sla.dense.LmSuffStat(row_ix, col_ix, n_offspring_flat, cxx_flat, cxy_flat)


def reverse_edges(ik: List[np.ndarray]) -> List[List[np.ndarray]]:

    return [[np.where(ik_ == i)[0] for i in range(max(ik_) + 1)] for ik_ in ik]
