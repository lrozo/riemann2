"""
Contains the tools for visualizing latent spaces
(including uncertainty and volume)
"""
from itertools import product
from typing import Tuple, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from riemannsquared.models.wrapped_gplvm import WrappedGPLVM

torch.set_default_dtype(torch.float64)


def _from_tensor_to_image(
    values: torch.Tensor, limits_x: Tuple[float], limits_y: Tuple[float], grid_size: int
) -> np.ndarray:
    """
    Grabs a tensor of size (grid_size*grid_size) and transforms
    it into an image of size (grid_size, grid_size) according to
    the limits provided.

    Parameters
    ----------

    - values (torch.Tensor of shape (grid_size ** 2,)): the tensor
      to convert into an image.

    - limits_x (Tuple[float]): the limits of the gridsize (min, max).

    - limits_y (Tuple[float]): the limits of the gridsize (min, max).

    - grid_size (int): the size of the grid/image.

    Returns
    -------

    - image (np.ndarray of shape (grid_size, grid_size)): the image
      version of the values provided, assuming they are in the same
      order as the grid below.

    """
    grid = torch.Tensor(
        [
            [x, y]
            for x, y in product(
                torch.linspace(*limits_x, grid_size),
                reversed(torch.linspace(*limits_y, grid_size)),
            )
        ]
    )

    z_to_values_map = {
        (z1.item(), z2.item()): values[i].item() for i, (z1, z2) in enumerate(grid)
    }
    image_of_values = np.zeros((grid_size, grid_size))

    for j, x in enumerate(torch.linspace(*limits_x, grid_size)):
        for i, y in enumerate(reversed(torch.linspace(*limits_y, grid_size))):
            image_of_values[i, j] = z_to_values_map[x.item(), y.item()]

    return image_of_values


def visualize_latent_space(
    ax: plt.Axes,
    wgplvm: WrappedGPLVM,
    plot_uncertainty: bool = True,
    plot_volume: bool = False,
    grid_size_in_latent_space: int = 25,
    limits: Tuple[float] = None,
    return_limits: bool = False,
    include_jacobian_of_exp: bool = True,
    scale_for_covariance_term: float = 1.0,
    max_volume = None,
    **plot_kwargs,
) -> Union[None, Tuple[float]]:
    """
    Plots the latent variables of a Wrapped GPLVM model in the
    given axis, dealing internally with the cases 1D vs. 2D vs. 3D.

    Parameters
    ----------

    - ax (plt.Axes) the axis in which to plot.

    - wgplvm: A wrapped GPLVM object. Remember to pass it in eval
      mode to get posterior estimates.

    - plot_uncertainty (bool, default=True) controls whether we plot
      a heatmap of uncertainty (in a grid of size specified in the
      "grid_size_in_latent_space" keyword argument).

    - plot_volume (bool, default=False) controls whether we plot
      a heatmap of pullback metric's volume (in a grid of size specified
      in the "grid_size_in_latent_space" keyword argument). When True,
      this will overwrite the visualization of the uncertainty, but I would
      recommend only one of these two (plot_uncertainty and plot_volume)
      to be True.

    - grid_size_in_latent_space (int): The size of the heatmap produced
      for plotting uncertainty or volume.

    - limits (Tuple[float], default=None): The limits you would like to specify
      for the grid (and therefore the entire plot) in latent space. By default,
      we compute these from the latent variables (plus/minus 5% padding).

    - return_limits (bool, default=False): When True, we return the computed (or
      provided) limits. This is useful for the interactive visualization we
      programmed in utils.visualization.play_with_model.

    - include_jacobian_of_exp (bool, default=True): When True, we include
      an approximation of the jacobian of the exponential map in the computation
      of the pullback metric.

    - max_volume (float, default=None): Upper limit on the metric volume. 
      This is used to improve the visualization of the metric volume.

    - **plot_kwargs: all the keyword arguments you would like to pass
      to the ax.scatter of the target points.

    Returns
    -------

    - limits (Tuple[float], optionally): the limits of the grid/latent space.
    """
    latent_variables = wgplvm.latent_variable()

    # Getting hyperparameters for the plots
    if limits is None:
        # x limit
        min_in_latent_x = latent_variables[:, 0].min().item()
        max_in_latent_x = latent_variables[:, 0].max().item()
        min_in_latent_y = latent_variables[:, 1].min().item()
        max_in_latent_y = latent_variables[:, 1].max().item()
        diff_in_x = abs(max_in_latent_x - min_in_latent_x)
        diff_in_y = abs(max_in_latent_y - min_in_latent_y)
        if diff_in_x > diff_in_y:
            min_in_latent_y -= (diff_in_x - diff_in_y) / 2.0
            max_in_latent_y += (diff_in_x - diff_in_y) / 2.0
        else:
            min_in_latent_x -= (diff_in_y - diff_in_x) / 2.0
            max_in_latent_x += (diff_in_y - diff_in_x) / 2.0
        padding = (max_in_latent_x - min_in_latent_x) * 0.05
        limits_x = (min_in_latent_x - padding, max_in_latent_x + padding)
        limits_y = (min_in_latent_y - padding, max_in_latent_y + padding)

    if len(latent_variables.shape) == 1 or latent_variables.shape[1] == 1:
        plot_ = "1D"
        x = latent_variables.flatten().detach().numpy()
        y = np.zeros_like(x)
    elif latent_variables.shape[1] == 2:
        plot_ = "2D"
        x = latent_variables[:, 0].detach().numpy()
        y = latent_variables[:, 1].detach().numpy()
    elif latent_variables.shape[1] == 3:
        plot_ = "3D"
        x = latent_variables[:, 0].detach().numpy()
        y = latent_variables[:, 1].detach().numpy()
        z = latent_variables[:, 2].detach().numpy()

    # Plotting the points
    if plot_ in ["1D", "2D"]:
        ax.scatter(x, y, c='steelblue', s=20, **plot_kwargs)
    else:
        ax.scatter(x, y, z, c='k', **plot_kwargs)

    # Plotting uncertainty in latent dim 2.
    if plot_ == "2D" and plot_uncertainty:
        fine_grid_in_latent_space = torch.Tensor(
            [
                [x, y]
                for x, y in product(
                    torch.linspace(*limits_x, grid_size_in_latent_space),
                    reversed(torch.linspace(*limits_y, grid_size_in_latent_space)),
                )
            ]
        )
        wrapped_gaussian_distribution_for_fine_grid = wgplvm(fine_grid_in_latent_space)
        variance = wrapped_gaussian_distribution_for_fine_grid.variance.sum(dim=1)

        img_of_variance = _from_tensor_to_image(
            variance, limits_x, limits_y, grid_size_in_latent_space
        )
        plot_ = ax.imshow(img_of_variance, cmap='magma', 
                          extent=[*limits_x, *limits_y], interpolation="bicubic")
        cbar = plt.colorbar(plot_, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=22)
        cbar.locator = MaxNLocator(nbins=5)

    if plot_ == "2D" and plot_volume:
        fine_grid_in_latent_space = torch.Tensor(
            [
                [x, y]
                for x, y in product(
                    torch.linspace(*limits_x, grid_size_in_latent_space),
                    reversed(torch.linspace(*limits_y, grid_size_in_latent_space)),
                )
            ]
        )
        pullback_metrics = wgplvm.predict_pullback_metric(
            fine_grid_in_latent_space,
            include_jacobian_of_exp=include_jacobian_of_exp,
            scale_for_covariance_term=scale_for_covariance_term,
        )
        volumes = torch.Tensor(
            [
                pullback_metric.det().sqrt().item()
                for pullback_metric in pullback_metrics
            ]
        )
        img_of_volumes = _from_tensor_to_image(
            volumes, limits_x, limits_y, grid_size_in_latent_space
        )
        if max_volume:
            img_of_volumes[img_of_volumes > max_volume] = max_volume
        plot_ = ax.imshow(img_of_volumes, cmap='magma',
                          extent=[*limits_x, *limits_y], interpolation="bicubic")
        cbar = plt.colorbar(plot_, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=22)
        cbar.locator = MaxNLocator(nbins=5)

    if return_limits:
        return (limits_x, limits_y)
