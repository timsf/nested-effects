from typing import Tuple, List

import numpy as np
import numpy.typing as npt

import nfx.bprop.dense
import nfx.sla.dense


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]


def eval_suff_stat(y: FloatArr, x: FloatArr) -> Tuple[int, FloatArr, FloatArr, FloatArr]:

    cyy = np.sum(np.square(y), 1)
    cxx = x.T @ x
    cxy = y @ x
    return y.shape[1], cyy, cxx, cxy


def process_bprop(dim_y: int, cyy: FloatArr, cxx: FloatArr, cxy: FloatArr) -> List[nfx.bprop.dense.LmSuffStat]:

    return [nfx.bprop.dense.LmSuffStat(cyy_, cxx, cxy_, dim_y) for cyy_, cxy_ in zip(cyy, cxy)]


def process_sla(dim_y: int, cxx: FloatArr, cxy: FloatArr, ik: List[IntArr], iik: List[List[IntArr]]
                ) -> nfx.sla.dense.LmSuffStat:

    n_offspring = [[len(iik__) for iik__ in iik_] for iik_ in iik] + [[len(iik[-1])]]
    n_offspring_flat = np.repeat(np.hstack(n_offspring), cxy.shape[1] ** 2)
    row_ix, col_ix = nfx.sla.dense.prepare_sparse_indices(ik, cxy.shape[1])
    cxy_flat = np.hstack(cxy)
    cxx_flat = np.tile(cxx.flatten(), len(ik[0]))
    return nfx.sla.dense.LmSuffStat(row_ix, col_ix, n_offspring_flat, cxx_flat, cxy_flat)


def reverse_edges(ik: List[IntArr]) -> List[List[IntArr]]:

    return [[np.where(ik_ == i)[0] for i in range(max(ik_) + 1)] for ik_ in ik]
