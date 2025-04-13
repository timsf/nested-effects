import numpy as np
import numpy.typing as npt


FloatArr = npt.NDArray[np.float64]


def swm_update(init: FloatArr, a: FloatArr, b: FloatArr) -> FloatArr:

    inv = init.copy()
    for a_, b_ in zip(a, b):
        inv -= np.outer(inv @ a_, inv.T @ b_) / (1 + b_ @ inv @ a_)
    return inv


def dual_inverse_update(a: FloatArr, b: FloatArr) -> FloatArr:

    return b @ np.linalg.inv(a + b) @ a
