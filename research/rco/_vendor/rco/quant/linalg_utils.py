# Vendored from IST-DASLab/RCO at 9a1e09c07d468109cbe60a1b87d5036034a79d10.
# Apache-2.0; see ../LICENSE and ../NOTICE.
import torch
from torch import Tensor


__all__ = ["inv_sym"]


def inv_sym(X: Tensor):
    """
    More efficient and stable inversion of symmetric matrices.
    """
    return torch.cholesky_inverse(torch.linalg.cholesky(X))
