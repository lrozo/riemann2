from typing import List

import matplotlib.pyplot as plt
import matplotlib.colors as pltc

import torch
from mayavi import mlab

from geomstats import visualization as manifold_visualization

from manifolds.hypersphere_vectorized import HypersphereVectorized

from metrics.sphere_metric_vectorized import SphereMetricVectorized

from riemannsquared.models.wrapped_gplvm import WrappedGPLVM

from utils.visualization.gp.tangent_vectors import plot_tangent_vectors_in_axis
from utils.visualization.manifolds.sphere_utils import plot_sphere_mayavi


def visualize_wrapped_gplvm_on_r2(
    axes: List[plt.Axes],
    wgplvm: WrappedGPLVM,
    plot_basepoint_function: bool = True,
    plot_tangent_vectors: bool = False,
    geodesic: torch.Tensor = None,
    **plot_kwargs
):
    """
    Visualizes a Wrapped GPLVM trained on the product manifold
    R2 x S2.

    Parameters
    ----------

    - axes (List[plt.Axes]): a list with two axes, one for R3 and
      another one for S2.

    - wgplvm (WrappedGPLVM): a Wrapped GPLVM on the product manifold
      R3xS2.

    - plot_basepoint_function (bool, default=True) controls whether
      we plot the basepoints as red diamonds.

    - plot_tangent_vectors (bool, default=False) controls whether we
      plot the tangent vectors as arrows on top of the sphere.
    """
    # Unpacking axes
    ax_for_r2 = axes

    # Plotting training data
    targets = wgplvm.training_targets
    ax_for_r2.scatter(targets[:, 0].detach().numpy(), targets[:, 1].detach().numpy(), c='royalblue', **plot_kwargs)

    # Plotting predictions at latent variables
    latent_variables = wgplvm.latent_variable()
    wrapped_gaussian_distribution = wgplvm(latent_variables)
    mean_in_product = wrapped_gaussian_distribution.mean
    ax_for_r2.scatter(mean_in_product[:, 0].detach().numpy(), mean_in_product[:, 1].detach().numpy(), c='blueviolet',
                      **plot_kwargs)

    # Plotting the prior (basepoints)
    if plot_basepoint_function:
        prior_kwargs = {"marker": "d", "color": "red", "alpha": 0.1}
        basepoint_prior = wgplvm.basepoint_function(latent_variables)
        ax_for_r2.scatter(
            basepoint_prior[:, 0].detach().numpy(),
            basepoint_prior[:, 1].detach().numpy(),
            **prior_kwargs
        )

    # Plotting the geodesic (if it was provided)
    if geodesic is not None:
        wgd_of_geodesics = wgplvm(geodesic)
        mean_geodesic = wgd_of_geodesics.mean
        ax_for_r2.plot(mean_geodesic[:, 0].detach().numpy(), mean_geodesic[:, 1].detach().numpy(), "-k", linewidth=2)


def visualize_wrapped_gplvm_on_s2_mayavi(fig_s2, wgplvm: WrappedGPLVM, traj_index,
                                            plot_basepoint_function: bool = True, geodesic: torch.Tensor = None):
    """
    Visualizes a Wrapped GPLVM trained on the product manifold R2 x S2 using Mayavi.

    Parameters
    ----------

    - axes (List[plt.Axes]): The axes for the Euclidean plot in R2 (in matplotlib).

    - wgplvm (WrappedGPLVM): a Wrapped GPLVM on the product manifold R3xS2.

    - plot_basepoint_function (bool, default=True) controls whether we plot the basepoints as red diamonds.

    - plot_tangent_vectors (bool, default=False) controls whether we plot the tangent vectors as arrows on the sphere.
    """
    # Plotting the sphere
    plot_sphere_mayavi(radius=1.0, opacity=0.2, figure=fig_s2)
    # TODO: We may move this as an argument to the function
    r, g, b = 18.0 / 255.0, 68.0 / 255.0, 112.0 / 255.0
    color_demo = (r, g, b)
    r, g, b = 18.0 / 255.0, 112.0 / 255.0, 90.0 / 255.0
    color_pred = (r, g, b)
    r, g, b = 112.0 / 255.0, 18.0 / 255.0, 18.0 / 255.0
    color_geo = (r, g, b)

    # Predictions at latent variables
    latent_variables = wgplvm.latent_variable()
    wrapped_gaussian_distribution = wgplvm(latent_variables)
    mean_in_product = wrapped_gaussian_distribution.mean

    # Plotting training data
    targets = wgplvm.training_targets
    # Get data as trajectories
    target_traj = targets.tensor_split(traj_index[:-1].type(torch. int64))
    predict_traj = mean_in_product.tensor_split(traj_index[:-1].type(torch. int64))

    # Plot in R2 and S2
    for n in range(target_traj.__len__()):
        mlab.plot3d(target_traj[n][:, 0], target_traj[n][:, 1], target_traj[n][:, 2], line_width=1.5, tube_radius=0.005,
                    color=color_demo, figure=fig_s2)
        mlab.plot3d(predict_traj[n][:, 0].detach().numpy(), predict_traj[n][:, 1].detach().numpy(),
                    predict_traj[n][:, 2].detach().numpy(), line_width=1.5, tube_radius=0.005, color=color_pred,
                    figure=fig_s2)

    # Plotting the prior (basepoints)
    if plot_basepoint_function:
        basepoint_prior = wgplvm.basepoint_function(latent_variables)
        mlab.points3d(basepoint_prior[:, 0].detach().numpy(), basepoint_prior[:, 1].detach().numpy(),
                      basepoint_prior[:, 2].detach().numpy(), color=(0.8, 0.1, 0.2), scale_factor=0.05, figure=fig_s2)

    # Plotting the geodesic (if it was provided)
    if geodesic is not None:
        wgd_of_geodesics = wgplvm(geodesic)
        mean_geodesic = wgd_of_geodesics.mean
        mlab.plot3d(mean_geodesic[:, 0].detach().numpy(), mean_geodesic[:, 1].detach().numpy(),
                    mean_geodesic[:, 2].detach().numpy(), line_width=1.5, tube_radius=0.005, color=color_geo,
                    figure=fig_s2)


def visualize_wrapped_gplvm_on_r2_s2(
    axes: List[plt.Axes],
    wgplvm: WrappedGPLVM,
    plot_basepoint_function: bool = True,
    plot_tangent_vectors: bool = False,
    geodesic: torch.Tensor = None,
    **plot_kwargs
):
    """
    Visualizes a Wrapped GPLVM trained on the product manifold
    R2 x S2.

    Parameters
    ----------

    - axes (List[plt.Axes]): a list with two axes, one for R3 and
      another one for S2.

    - wgplvm (WrappedGPLVM): a Wrapped GPLVM on the product manifold
      R3xS2.

    - plot_basepoint_function (bool, default=True) controls whether
      we plot the basepoints as red diamonds.

    - plot_tangent_vectors (bool, default=False) controls whether we
      plot the tangent vectors as arrows on top of the sphere.
    """
    # Unpacking axes
    ax_for_r2, ax_for_s2 = axes

    # Plotting the sphere
    sphere_viz = manifold_visualization.Sphere()
    sphere_viz.draw(ax_for_s2)

    # Plotting training data
    targets = wgplvm.training_targets
    ax_for_r2.scatter(targets[:, 0].detach().numpy(), targets[:, 1].detach().numpy(), c='royalblue', **plot_kwargs)
    ax_for_s2.scatter(targets[:, 2].detach().numpy(), targets[:, 3].detach().numpy(), targets[:, 4].detach().numpy(),
                      c='royalblue', **plot_kwargs)

    # Plotting predictions at latent variables
    latent_variables = wgplvm.latent_variable()
    wrapped_gaussian_distribution = wgplvm(latent_variables)
    mean_in_product = wrapped_gaussian_distribution.mean
    ax_for_r2.scatter(mean_in_product[:, 0].detach().numpy(), mean_in_product[:, 1].detach().numpy(), c='blueviolet',
                      **plot_kwargs)
    ax_for_s2.scatter(mean_in_product[:, 2].detach().numpy(), mean_in_product[:, 3].detach().numpy(),
                      mean_in_product[:, 4].detach().numpy(), c='blueviolet', **plot_kwargs)

    # Plotting the prior (basepoints)
    if plot_basepoint_function:
        prior_kwargs = {"marker": "d", "color": "red", "alpha": 0.1}
        basepoint_prior = wgplvm.basepoint_function(latent_variables)
        ax_for_r2.scatter(
            basepoint_prior[:, 0].detach().numpy(),
            basepoint_prior[:, 1].detach().numpy(),
            **prior_kwargs
        )
        ax_for_s2.scatter(
            basepoint_prior[:, 2].detach().numpy(),
            basepoint_prior[:, 3].detach().numpy(),
            basepoint_prior[:, 4].detach().numpy(),
            **prior_kwargs
        )

    # Plotting the geodesic (if it was provided)
    if geodesic is not None:
        wgd_of_geodesics = wgplvm(geodesic)
        mean_geodesic = wgd_of_geodesics.mean
        ax_for_r2.plot(mean_geodesic[:, 0].detach().numpy(), mean_geodesic[:, 1].detach().numpy(), "-k", linewidth=2)
        ax_for_s2.plot(mean_geodesic[:, 2].detach().numpy(), mean_geodesic[:, 3].detach().numpy(),
                       mean_geodesic[:, 4].detach().numpy(), "-k", linewidth=2)

    if plot_tangent_vectors:
        basepoints = wgplvm.basepoint_function(latent_variables)[:, 2:]

        s2_manifold = HypersphereVectorized(2)
        sphere_metric = SphereMetricVectorized(s2_manifold)
        tangent_targets = wgplvm.get_tangent_targets(
            latent_variables, wrapped_gaussian_distribution.mean
        )[:, 2:]
        tangent_targets = sphere_metric._from_coordinates_to_tangent_vec(
            tangent_targets, basepoints
        )

        plot_tangent_vectors_in_axis(ax_for_s2, basepoints, tangent_targets)


def visualize_wrapped_gplvm_on_r2_s2_mayavi(axes_r2: plt.Axes, fig_s2, wgplvm: WrappedGPLVM, traj_index,
                                            plot_basepoint_function: bool = True, 
                                            plot_reconstructions: bool = False,
                                            geodesic: torch.Tensor = None, line: torch.Tensor = None,
                                            d_geodesics: List = None):
    """
    Visualizes a Wrapped GPLVM trained on the product manifold R2 x S2 using Mayavi.

    Parameters
    ----------

    - axes (List[plt.Axes]): The axes for the Euclidean plot in R2 (in matplotlib).

    - wgplvm (WrappedGPLVM): a Wrapped GPLVM on the product manifold R3xS2.

    - plot_basepoint_function (bool, default=True) controls whether we plot the basepoints as red diamonds.

    - plot_tangent_vectors (bool, default=False) controls whether we plot the tangent vectors as arrows on the sphere.
    """
    # Plotting the sphere
    plot_sphere_mayavi(radius=1.0, opacity=0.2, figure=fig_s2)
    # TODO: We may move this as an argument to the function
    r, g, b = 18.0 / 255.0, 68.0 / 255.0, 112.0 / 255.0
    color_demo = (r, g, b)
    color_demo = pltc.to_rgb('black')
    r, g, b = 18.0 / 255.0, 112.0 / 255.0, 90.0 / 255.0
    color_pred = pltc.to_rgb('steelblue')
    r, g, b = 112.0 / 255.0, 18.0 / 255.0, 18.0 / 255.0
    color_geo = pltc.to_rgb('orange')
    color_line = pltc.to_rgb('crimson')
    color_d_geo = pltc.to_rgb('maroon')

    # Predictions at latent variables
    latent_variables = wgplvm.latent_variable()
    wrapped_gaussian_distribution = wgplvm(latent_variables)
    mean_in_product = wrapped_gaussian_distribution.mean

    # Plotting training data
    targets = wgplvm.training_targets
    # Get data as trajectories
    target_traj = targets.tensor_split(traj_index[:-1].type(torch. int64))
    predict_traj = mean_in_product.tensor_split(traj_index[:-1].type(torch. int64))

    # Plot in R2 and S2
    for n in range(target_traj.__len__()):
        axes_r2.plot(target_traj[n][:, 0], target_traj[n][:, 1], linewidth=3.0, color=color_demo)
        mlab.plot3d(target_traj[n][:, 2], target_traj[n][:, 3], target_traj[n][:, 4], line_width=8.0, tube_radius=None,
                    color=color_demo, figure=fig_s2)
        if plot_reconstructions:
          axes_r2.plot(predict_traj[n][:, 0].detach().numpy(), predict_traj[n][:, 1].detach().numpy(),
                      linewidth=2.0, color=color_pred)
          mlab.plot3d(predict_traj[n][:, 2].detach().numpy(), predict_traj[n][:, 3].detach().numpy(),
                      predict_traj[n][:, 4].detach().numpy(), line_width=7.0, tube_radius=None, color=color_pred,
                      figure=fig_s2)

    # Plotting the prior (basepoints)
    if plot_basepoint_function:
        basepoint_prior = wgplvm.basepoint_function(latent_variables)
        # axes_r2.scatter(basepoint_prior[:, 0].detach().numpy(), basepoint_prior[:, 1].detach().numpy(),
        #                 marker="d", color="red", alpha=0.1)
        mlab.points3d(basepoint_prior[:, 2].detach().numpy(), basepoint_prior[:, 3].detach().numpy(),
                      basepoint_prior[:, 4].detach().numpy(), color=pltc.to_rgb('navy'), scale_factor=0.03, figure=fig_s2)

    # Plotting the geodesic (if it was provided)
    if geodesic is not None:
        wgd_of_geodesics = wgplvm(geodesic)
        mean_geodesic = wgd_of_geodesics.mean 
        axes_r2.plot(mean_geodesic[:, 0].detach().numpy(), mean_geodesic[:, 1].detach().numpy(),
                     linewidth=4.0, color=color_geo)
        mlab.plot3d(mean_geodesic[:, 2].detach().numpy(), mean_geodesic[:, 3].detach().numpy(),
                    mean_geodesic[:, 4].detach().numpy(), line_width=12.0, tube_radius=None, color=color_geo,
                    figure=fig_s2)
    # Plotting the linear interpolation (if it was provided)
    if line is not None:
        wgd_of_line = wgplvm(line)
        mean_line = wgd_of_line.mean
        axes_r2.plot(mean_line[:, 0].detach().numpy(), mean_line[:, 1].detach().numpy(),
                     linewidth=4.0, color=color_line)
        mlab.plot3d(mean_line[:, 2].detach().numpy(), mean_line[:, 3].detach().numpy(),
                    mean_line[:, 4].detach().numpy(), line_width=11.0, tube_radius=None, color=color_line,
                    figure=fig_s2)
    # Plotting the density metric geodesic (if it was provided)
    if d_geodesics is not None:
        for d_geodesic in d_geodesics:
          wgd_of_d_geodesics = wgplvm(d_geodesic)
          mean_d_geodesic = wgd_of_d_geodesics.mean 
          axes_r2.plot(mean_d_geodesic[:, 0].detach().numpy(), mean_d_geodesic[:, 1].detach().numpy(),
                      linewidth=4.0, color=color_d_geo)
          mlab.plot3d(mean_d_geodesic[:, 2].detach().numpy(), mean_d_geodesic[:, 3].detach().numpy(),
                      mean_d_geodesic[:, 4].detach().numpy(), line_width=12.0, tube_radius=None, color=color_d_geo,
                      figure=fig_s2)

