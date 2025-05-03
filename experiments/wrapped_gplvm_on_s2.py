"""
This script implements the toy experiment: learning a
(generalized) wrapped GPLVM of letter-shaped data that
lies in S2. It is ideal for visualization and for getting a basic 
understanding of what our model does.

A similar example can be found under example/wrapped_gp_on_r3_s2.py

We use a Wrapped GPLVM, where the underlying Euclidean GP on the
tangent bundle/space is a multitask GP (i.e. we model the correlation
between different dimensions). or more information about WGPLVMs, 
check the implementation under riemannsquared.models.wrapped_gplvm.

"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from mayavi import mlab

import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.kernels import ScaleKernel, MultitaskKernel, RBFKernel

from geomstats.geometry.euclidean import Euclidean

from manifolds.hypersphere_vectorized import HypersphereVectorized
from manifolds.product_manifold_vectorized import ProductManifoldVectorized

from metrics.euclidean import AugmentedCanonicalEuclideanMetric

from kernels.kernels_sphere import SphereRiemannianGaussianKernel

from utils.data.load_letter_manifolds import load_toy_example_data_v2
from riemannsquared.training.gplvm import train_wrapped_gplvm
from utils.visualization.gplvm.latent_space import visualize_latent_space
from utils.visualization.gplvm.r2_s2 import visualize_wrapped_gplvm_on_s2_mayavi
from utils.wrappers.stochman_manifolds import DiscreteGPLVMManifoldWrapper, GPLVMManifoldWrapper

from riemannsquared.models.wrapped_gplvm import WrappedGPLVM, BackConstrainedWrappedGPLVM

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()


def load_wgplvm_on_toy_experiment() -> Tuple[WrappedGPLVM, torch.Tensor, torch.Tensor]:
    """
    Returns a Wrapped GPLVM alongside the training and test
    data in a deterministic way. This way, we have a reliable way of
    instantiating the model (keeping the same hyperparameters and
    train-test split.)

    Returns
    -------

    - WrappedGPLVM (ExactEuclideanGPLVM): an Exact Euclidean GPLVM
      instantiated on the toy experiment dataset.

    - training_data (torch.Tensor): an 80% sample of the original
      dataset, used for training.

    - test_data (torch.Tensor): the remaining 20% of the original
      dataset, used for testing.
    """
    # Hardcoding some hyperparameters
    # Dim. of latent space (we have only tried 2)
    latent_dim = 2

    # Whether to split into train/test, and test percentage
    split_into_train_test = True
    test_percentage = 0.2

    # A subsample rate, to consider less data. (1 == all dataset)
    subsample_rate = 1

    # Noise that is synthetically added to the data.
    noise_level_in_data = 0.0

    # Random state to have the train/test split be deterministic.
    random_state_for_split = 420

    # Priors for likelihood noise, and kernel length scale and output scale.
    # (They need to be gpytorch priors)
    noise_prior = None
    lengthscale_prior = None  # gpytorch.priors.GammaPrior(10.0, 1.0)
    outputscale_prior = None

    # Defining the manifold.
    dim_euclidean = 2
    dim_sphere = 2
    s2_manifold = HypersphereVectorized(dim_sphere)

    # Loading up the training/testing data
    toy_example_training_data, toy_example_test_data, trajectory_indexes = load_toy_example_data_v2(
        noise_level=noise_level_in_data, split_into_train_test=split_into_train_test, scale_factor_r2=1.0)

    # Subsampling the training data (to have a smaller dataset)
    toy_example_training_data = toy_example_training_data[::subsample_rate]
    toy_example_test_data = toy_example_test_data[::subsample_rate]

    # Take only spherical part of the data
    toy_example_training_data = toy_example_training_data[:, 2:]
    toy_example_test_data = toy_example_test_data[:, 2:]

    # Defining the basepoint function: constant on (the origin, north pole).
    basepoint = torch.Tensor([0.0, 0.0, 1.0])
    basepoint_function = lambda x: np.repeat(basepoint.unsqueeze(0), x.shape[0], axis=0)

    # Defining the likelihood and kernel in the tangent space
    # of R2xS2. Notice how this tangent space is of dimension 4.
    tangent_space_dim = s2_manifold.metric.tangent_space_dim

    # The tangent space likelihood
    tangent_space_likelihood = MultitaskGaussianLikelihood(num_tasks=tangent_space_dim,
                                                           noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
                                                           noise_prior=noise_prior)

    # Setting up a multitask kernel with a low-mean prior for
    # the lengthscale.
    kernel = MultitaskKernel(ScaleKernel(RBFKernel(lengthscale_prior=lengthscale_prior),  # (ard_num_dims=latent_dim),  # Commented out to simplify training
                                         outputscale_prior=outputscale_prior), 
                                         num_tasks=tangent_space_dim,
                             rank=tangent_space_dim)

    # Creating the exact wrapped model
    # wrapped_gplvm_on_r2_s2 = WrappedGPLVM(latent_dim, product_manifold_r2_s2, basepoint_function,
    #                                       toy_example_training_data, tangent_space_likelihood, tangent_kernel=kernel,
    #                                       trajectory_indexes=trajectory_indexes, initialization="stress")

    # Creating the exact wrapped model with back constraints
    k_fct_sphere = SphereRiemannianGaussianKernel(dim_sphere)

    targets_kernel = k_fct_sphere
    targets_kernel.lengthscale = 0.7
    wrapped_gplvm_on_s2 = BackConstrainedWrappedGPLVM(latent_dim, s2_manifold, basepoint_function,
                                                         toy_example_training_data, tangent_space_likelihood,
                                                         tangent_kernel=kernel, trajectory_indexes=trajectory_indexes,
                                                         initialization="pca",
                                                         targets_kernel=targets_kernel)

    return wrapped_gplvm_on_s2, toy_example_training_data, toy_example_test_data, trajectory_indexes


def toy_experiment_on_letter_manifolds():
    """
    The main function. Here we train and visualize a WrappedGPLVM on
    the product of letter manifolds on R2 x S2.
    """
    # Training hyperparameters:
    exp_name = "wrapped_gplvm_S2_on_toy_experiment"
    max_iterations = 200
    learning_rate = 0.05
    load_saved = True
    discrete_geodesic = True

    # Visualization hyperparameters:
    # We go for low resolution (10) for speed in the iteration process.
    uncertainty_and_volume_grid_size = 50
    discretized_grid_size = 50
    projection_grid_size = 75
    max_volume = None  # TODO add to visualization

    # Load the model specified above.
    wrapped_gplvm_on_s2, _, test_data, data_indexes = load_wgplvm_on_toy_experiment()

    # Plot initial latent variables
    fig = plt.figure(figsize=(7, 7))
    fig_ax = fig.gca()
    print("Visualizing latent space")
    visualize_latent_space(
        fig_ax,
        wrapped_gplvm_on_s2,
        grid_size_in_latent_space=uncertainty_and_volume_grid_size,
        plot_uncertainty=False
    )
    fig_ax.axis('equal')
    plt.show()
    print(f'Lengthscale value value = {wrapped_gplvm_on_s2.tangent_kernel.data_covar_module.base_kernel.lengthscale}')
    print(f'Outputscale value value = {wrapped_gplvm_on_s2.tangent_kernel.data_covar_module.outputscale}')
    print(f'Back constraints weight value = {wrapped_gplvm_on_s2.latent_variable.weights.data}')
    print(f'Latent variables = {wrapped_gplvm_on_s2.latent_variable()}')

    # Train the model.
    wrapped_gplvm_on_s2 = train_wrapped_gplvm(wrapped_gplvm_on_s2, verbose=True, save_as=exp_name,
                                                 max_iterations=max_iterations, learning_rate=learning_rate,
                                                 load_saved=load_saved)

    # ------------------------ Visualizing ---------------------------
    # This visualization is temporal, for quickly assessing the quality of a given training run.
    # For the final version of the plots you can check the ../results folder.
    wrapped_gplvm_on_s2.eval()

    # Print some of the model parameters
    print(f'Lengthscale value value = {wrapped_gplvm_on_s2.tangent_kernel.data_covar_module.base_kernel.lengthscale}')
    print(f'Outputscale value value = {wrapped_gplvm_on_s2.tangent_kernel.data_covar_module.outputscale}')
    print(f'Back constraints weight value = {wrapped_gplvm_on_s2.latent_variable.weights.data}')
    print(f'Latent variables = {wrapped_gplvm_on_s2.latent_variable()}')

    # Check training error
    posterior = wrapped_gplvm_on_s2(wrapped_gplvm_on_s2.latent_variable())
    error = (posterior.mean - wrapped_gplvm_on_s2.training_targets)
    print("Mean absolute error: ", torch.mean(torch.abs(error)))

    # Projecting test data onto the latent space and plotting the projections
    print("Projecting test data onto latent space via back constraints")
    projected_test_data = wrapped_gplvm_on_s2.latent_variable.back_constraint_function(test_data)

    # Create plots folder
    plots_path = "/plots/wrappedGPLVM_S2/"
    if not os.path.exists(ROOT_DIR.as_posix() + plots_path):
        print("Creating new plots folder for the experiment...")
        os.makedirs(ROOT_DIR.as_posix() + plots_path)

    # Creating the figure and axes
    fig1 = plt.figure(figsize=(7, 7))
    fig1_ax = fig1.gca()
    # Just plot the learned embeddings
    print("Simple visualization of learned embeddings.")
    visualize_latent_space(fig1_ax, wrapped_gplvm_on_s2, grid_size_in_latent_space=uncertainty_and_volume_grid_size,
                           plot_uncertainty=False)
    fig1_ax.axis("off")
    fig1.tight_layout()
    fig1.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_LatentSpace.png")

    # Latent space, latent test data (BC) and GP uncertainty
    print("Visualizing latent space (with GP uncertainty)")
    visualize_latent_space(fig1_ax, wrapped_gplvm_on_s2, grid_size_in_latent_space=uncertainty_and_volume_grid_size)
    #  and plotting the projections
    fig1_ax.scatter(projected_test_data[:, 0].detach().numpy(), projected_test_data[:, 1].detach().numpy(), c="r",
                    marker="d", alpha=0.1)
    fig1_ax.axis("off")
    fig1.tight_layout()

    # # Visualizing latent space with volume
    # fig2 = plt.figure(figsize=(7, 7))
    # fig2_ax = fig2.gca()
    # limits = visualize_latent_space(fig2_ax, wrapped_gplvm_on_s2, plot_uncertainty=False, plot_volume=True,
    #                                 grid_size_in_latent_space=uncertainty_and_volume_grid_size, return_limits=True)
    # fig2_ax.axis("off")
    # fig2.tight_layout()

    # # # Compute geodesics
    # # if discrete_geodesic:
    # #     print("Computing discrete geodesics")
    # #     wgpvlm_as_manifold = DiscreteGPLVMManifoldWrapper(wrapped_gplvm_on_s2,
    # #                                                       [torch.linspace(*limits[0], discretized_grid_size),
    # #                                                        torch.linspace(*limits[1], discretized_grid_size)])
    # #     wgpvlm_as_manifold.fit()
    # # else:
    # #     print("Computing continuous geodesics")
    # #     wgpvlm_as_manifold = GPLVMManifoldWrapper(wrapped_gplvm_on_s2)
    # # # Compute geodesic
    # # geodesic, _ = wgpvlm_as_manifold.connecting_geodesic(wrapped_gplvm_on_s2.latent_variable()[0],
    # #                                                      wrapped_gplvm_on_s2.latent_variable()[-1])
    # # time = torch.linspace(0, 1, 50)
    # # # Visualize them in latent space.
    # # geodesic_points = geodesic(time).detach().numpy()
    # # fig1_ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], "-k", linewidth=2)
    # # fig2_ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], "-k", linewidth=2)
    # # Save latent space plots
    # fig1.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_LatentSpaceUncertainty.png")
    # fig2.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_LatentSpaceMetricVol.png")

    # Plot the decoded predictions and geodesics on the ambient manifolds
    print("Plotting the manifold predictions")
    # Sphere plot S2 with mayavi
    figs2 = mlab.figure(bgcolor=(1.0, 1.0, 1.0))
    mlab.view(azimuth=0, elevation=180)
    visualize_wrapped_gplvm_on_s2_mayavi(fig_s2=figs2, wgplvm=wrapped_gplvm_on_s2,
                                            traj_index=data_indexes)  #, geodesic=geodesic(time))

    # fig_ax_s2.axis("off")  # Uncomment if using matplotlib
    # figs2.tight_layout()  # Uncomment if using matplotlib
    # figr2.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_R2.png")
    # figs2.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_S2.png") # Uncomment if using matplotlib

    # Sphere plot S2 with mayavi (uncomment lines below if using Mayavi)
    figs2.scene.camera.zoom(1.5)
    mlab.show()
    mlab.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_S2.png", magnification=2)

    plt.show()
    plt.close()


if __name__ == "__main__":
    toy_experiment_on_letter_manifolds()
