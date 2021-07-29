import numpy as np


def swm_update(init: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:

    inv = init.copy()
    for a_, b_ in zip(a, b):
        inv -= np.outer(inv @ a_, inv.T @ b_) / (1 + b_ @ inv @ a_)
    return inv


def dual_inverse_update(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    return b @ np.linalg.inv(a + b) @ a
