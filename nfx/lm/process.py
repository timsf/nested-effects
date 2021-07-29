from typing import Tuple, List

import numpy as np

import nfx.bprop.dense
import nfx.sla.dense


def eval_suff_stat(y: np.ndarray, x: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:

    cyy = np.sum(np.square(y), 1)
    cxx = x.T @ x
    cxy = y @ x
    return y.shape[1], cyy, cxx, cxy


def process_bprop(dim_y: int, cyy: np.ndarray, cxx: np.ndarray, cxy: np.ndarray) -> List[nfx.bprop.dense.LmSuffStat]:

    return [nfx.bprop.dense.LmSuffStat(cyy_, cxx, cxy_, dim_y) for cyy_, cxy_ in zip(cyy, cxy)]


def process_sla(dim_y: int, cxx: np.ndarray, cxy: np.ndarray, ik: np.ndarray, iik: List[np.ndarray]) -> nfx.sla.dense.LmSuffStat:

    n_offspring = [[len(iik__) for iik__ in iik_] for iik_ in iik] + [[len(iik[-1])]]
    n_offspring_flat = np.repeat(np.hstack(n_offspring), cxy.shape[1] ** 2)
    row_ix, col_ix = nfx.sla.dense.prepare_sparse_indices(ik, cxy.shape[1])
    cxy_flat = np.hstack(cxy)
    cxx_flat = np.tile(cxx.flatten(), len(ik[0]))
    return nfx.sla.dense.LmSuffStat(row_ix, col_ix, n_offspring_flat, cxx_flat, cxy_flat)


def reverse_edges(ik: List[np.ndarray]) -> List[List[np.ndarray]]:

    return [[np.where(ik_ == i)[0] for i in range(max(ik_) + 1)] for ik_ in ik]
