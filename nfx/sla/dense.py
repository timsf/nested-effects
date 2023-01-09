from typing import List, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt

import scipy.linalg
import scipy.sparse
import sksparse.cholmod


IntArr = npt.NDArray[np.int_]
FloatArr = npt.NDArray[np.float_]


class LmSuffStat(NamedTuple):
    row_ix: FloatArr
    col_ix: FloatArr
    n_offspring: FloatArr
    cxx: FloatArr
    cxy: FloatArr


def sample_nested_lm(
    y: LmSuffStat,
    ik: List[IntArr],
    mu0: FloatArr,
    tau0: FloatArr,
    tau: List[FloatArr],
    lam: FloatArr,
    ome: np.random.Generator,
) -> List[FloatArr]:

    jk = [len(ik_) for ik_ in ik] + [np.max(ik[-1]) + 1, 1]

    prec_flat = scipy.sparse.csc_matrix(fill_precision(y, ik, tau0, tau, lam))
    lcf_prec_flat = sksparse.cholmod.cholesky(prec_flat, ordering_method='natural')
    prior_weight_flat = np.hstack([np.zeros(prec_flat.shape[0] - len(mu0)), tau0 @ mu0])
    post_weight_flat = np.hstack([np.repeat(lam, len(mu0)) * y.cxy, np.zeros(len(prior_weight_flat) - len(y.cxy))])
    z_flat = ome.standard_normal(len(prior_weight_flat))
    bet_flat = lcf_prec_flat.solve_Lt(lcf_prec_flat.solve_L(post_weight_flat + prior_weight_flat, False) + z_flat, False)

    bet = [np.reshape(bet_, (jk_, len(mu0))) for jk_, bet_
           in zip(jk, np.split(bet_flat, np.cumsum(jk[:-1]) * len(mu0)))]
    return bet


def prepare_sparse_indices(ik: List[IntArr], dim: int) -> Tuple[IntArr, IntArr]:

    jk = [len(ik_) for ik_ in ik] + [np.max(ik[-1]) + 1, 1]

    on_row_block_ix = np.arange(sum(jk))
    block_offsets = np.hstack([ik_ + jk_ for ik_, jk_
                               in zip(ik + [np.int_(np.zeros(max(ik[-1] + 1)))], np.cumsum(jk))])
    off_row_block_ix = block_offsets
    off_col_block_ix = on_row_block_ix[:-1]

    on_row_ix = np.hstack([ix_ * dim + np.tile(np.arange(dim), dim) for ix_ in on_row_block_ix])
    on_col_ix = np.hstack([ix_ * dim + np.repeat(np.arange(dim), dim) for ix_ in on_row_block_ix])
    off_row_ix = np.hstack([ix_ * dim + np.tile(np.arange(dim), dim) for ix_ in off_row_block_ix])
    off_col_ix = np.hstack([ix_ * dim + np.repeat(np.arange(dim), dim) for ix_ in off_col_block_ix])

    row_ix = np.hstack([on_row_ix, off_row_ix])
    col_ix = np.hstack([on_col_ix, off_col_ix])
    return row_ix, col_ix


def fill_precision(
    y: LmSuffStat,
    ik: List[IntArr],
    tau0: FloatArr,
    tau: List[FloatArr],
    lam: FloatArr,
) -> scipy.sparse.coo_matrix:

    jk = [len(ik_) for ik_ in ik] + [np.max(ik[-1]) + 1, 1]

    values_on_diag_1 = np.hstack([np.tile(tau_.flatten(), jk_) for tau_, jk_ in zip(tau + [tau0], jk)])
    values_on_diag_2 = np.hstack([np.repeat(lam, np.prod(tau0.shape)) * y.cxx,
        y.n_offspring * np.hstack([np.tile(tau_.flatten(), jk_) for tau_, jk_ in zip(tau, jk[1:])])])

    values_on_diag = values_on_diag_1 + values_on_diag_2
    values_off_diag = -values_on_diag_1[:-np.prod(tau0.shape)]
    values = np.hstack([values_on_diag, values_off_diag])

    prec_flat = scipy.sparse.coo_matrix((values, (y.row_ix, y.col_ix)), 2 * (sum(jk) * tau0.shape[0],))
    return prec_flat
