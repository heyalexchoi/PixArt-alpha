"""
Adapted from https://github.com/google-research/google-research/blob/master/cmmd/distance.py
opted for Claude-based adaptation over this: https://github.com/sayakpaul/cmmd-pytorch/blob/main/distance.py
Saw unusual adaptation decisions in the latter.
https://arxiv.org/abs/2401.09603
"""
"""Memory-efficient MMD implementation in PyTorch."""

import torch

# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 1000


def compute_mmd(x, y) -> torch.Tensor:
    """Memory-efficient MMD implementation in PyTorch.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Args:
        x: The first set of embeddings of shape (n, embedding_dim).
        y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
        The MMD distance between x and y embedding sets.
    """
    # x = torch.tensor(x)
    # y = torch.tensor(y)

    # torch.matmul(x, x.T) etc. are not cached to avoid OOM when x has many rows.
    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(x, x.T)
                + x_sqnorms.unsqueeze(1)
                + x_sqnorms.unsqueeze(0)
            )
        )
    )
    k_xy = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(x, y.T)
                + x_sqnorms.unsqueeze(1)
                + y_sqnorms.unsqueeze(0)
            )
        )
    )
    k_yy = torch.mean(
        torch.exp(
            -gamma
            * (
                -2 * torch.matmul(y, y.T)
                + y_sqnorms.unsqueeze(1)
                + y_sqnorms.unsqueeze(0)
            )
        )
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)