"""
This module contains a simple, generic function
for plotting tangent vectors given basepoints and
the tangent vector values.
"""

from matplotlib import pyplot as plt
import torch


def plot_tangent_vectors_in_axis(
    ax: plt.Axes, basepoint: torch.Tensor, tangent_vectors: torch.Tensor
):
    """
    Plots the tangent vectors as arrows based on the provided
    basepoints, and ending in the provied tangent_vectors values.

    Parameters
    ----------

    - ax (plt.Axes): the axis in which to plot the arrows.

    - basepoint (torch.Tensor of shape (d,) or (b, d) for
      d in [2, 3]): the points from which the arrows should start.

    - tangent_vectors (torch.Tensor of shape (d,) or (b, d) for
      d in [2, 3]): the end-points of the arrows.
    """
    # Deal with shapes
    if len(basepoint.shape) == 1:
        if basepoint.shape[0] not in [2, 3]:
            raise ValueError(...)
        else:
            basepoint = basepoint.unsqueeze(0)

    if len(tangent_vectors.shape) == 1:
        if tangent_vectors.shape[0] not in [2, 3]:
            raise ValueError(...)
        else:
            tangent_vectors = tangent_vectors.unsqueeze(0)

    # At this point, both tensors have a batch shape.

    # Plotting the arrows in 2d and 3d separately.
    if basepoint.shape[1] == tangent_vectors.shape[1] == 3:
        ax.scatter(
            basepoint[:, 0].detach().numpy(),
            basepoint[:, 1].detach().numpy(),
            basepoint[:, 2].detach().numpy(),
            marker="d",
            s=15,
        )

        ax.quiver(
            basepoint[:, 0].detach().numpy(),
            basepoint[:, 1].detach().numpy(),
            basepoint[:, 2].detach().numpy(),
            tangent_vectors[:, 0].detach().numpy(),
            tangent_vectors[:, 1].detach().numpy(),
            tangent_vectors[:, 2].detach().numpy(),
            arrow_length_ratio=0.05,
            alpha=0.4,
        )
    elif basepoint.shape[1] == tangent_vectors.shape[1] == 2:
        ax.scatter(
            basepoint[:, 0].detach().numpy(),
            basepoint[:, 1].detach().numpy(),
            marker="d",
            s=15,
        )

        ax.quiver(
            basepoint[:, 0].detach().numpy(),
            basepoint[:, 1].detach().numpy(),
            tangent_vectors[:, 0].detach().numpy(),
            tangent_vectors[:, 1].detach().numpy(),
            arrow_length_ratio=0.05,
            alpha=0.4,
        )
    else:
        raise ValueError(...)
