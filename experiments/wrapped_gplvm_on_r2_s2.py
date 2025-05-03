"""
This script implements the toy experiment: learning a
(generalized) wrapped GPLVM of letter-shaped data that
lies in the product R2 x S2. It is ideal for visualization
and for getting a basic understanding of what our model does.
More information about this experiment can be found in Experiments
Section  of our paper.

We use a Wrapped GPLVM, where the underlying Euclidean GP on the
tangent bundle/space is a multitask GP (i.e. we model the correlation
between different dimensions). For more information about WGPLVMs, 
check the implementation under riemannsquared.models.wrapped_gplvm.

"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from mayavi import mlab
import numpy as np

import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
from gpytorch.kernels import ScaleKernel, MultitaskKernel, RBFKernel

from geomstats.geometry.euclidean import Euclidean

from manifolds.hypersphere_vectorized import HypersphereVectorized
from manifolds.product_manifold_vectorized import ProductManifoldVectorized

from metrics.euclidean import AugmentedCanonicalEuclideanMetric

from kernels.kernels_sphere import SphereRiemannianGaussianKernel

from utils.data.load_letter_manifolds import load_toy_example_data_v2
from riemannsquared.training.gplvm import train_wrapped_gplvm
from utils.visualization.gplvm.latent_space import visualize_latent_space
from utils.visualization.gplvm.r2_s2 import visualize_wrapped_gplvm_on_r2_s2_mayavi
from utils.wrappers.stochman_manifolds import DiscreteGPLVMManifoldWrapper, GPLVMManifoldWrapper, \
  geodesic_minimizing_energy, DensityMetricWrapper
from utils.analysis.dtwd import get_dynamic_time_warping_distance_list

from riemannsquared.models.wrapped_gplvm import WrappedGPLVM, BackConstrainedWrappedGPLVM
from riemannsquared.models.gpdm_prior import GPDMPrior


ROOT_DIR = Path(__file__).parent.parent.resolve()


def load_wgplvm_on_toy_experiment(model_name) -> Tuple[WrappedGPLVM, torch.Tensor, torch.Tensor]:
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

    # A subsample rate, to consider less data. (1 == all dataset)
    subsample_rate = 1

    # Noise that is synthetically added to the data.
    noise_level_in_data = 0.0

    # Priors for likelihood noise, and kernel length scale and output scale.
    # (They need to be gpytorch priors)
    noise_prior = None
    lengthscale_prior = None  # gpytorch.priors.GammaPrior(10.0, 1.0)
    outputscale_prior = None

    # Defining the manifold.
    dim_euclidean = 2
    dim_sphere = 2
    r2_manifold = Euclidean(dim_euclidean, equip=False)
    r2_manifold.equip_with_metric(AugmentedCanonicalEuclideanMetric)
    s2_manifold = HypersphereVectorized(dim_sphere)
    product_manifold_r2_s2 = ProductManifoldVectorized([r2_manifold, s2_manifold]) 

    # Loading up the training/testing data
    toy_example_training_data, toy_example_test_data, trajectory_indexes = load_toy_example_data_v2(
        noise_level=noise_level_in_data, split_into_train_test=split_into_train_test, scale_factor_r2=1.0)

    # Subsampling the training data (to have a smaller dataset)
    toy_example_training_data = toy_example_training_data[::subsample_rate]
    toy_example_test_data = toy_example_test_data[::subsample_rate]

    # Defining the basepoint function: constant on (the origin, north pole).
    basepoint = torch.Tensor([0.0, 0.0] + [0.0, 0.0, 1.0])
    basepoint_function = lambda x: np.repeat(basepoint.unsqueeze(0), x.shape[0], axis=0)

    # Defining the likelihood and kernel in the tangent space
    # of R2xS2. Notice how this tangent space is of dimension 4.
    tangent_space_dim = product_manifold_r2_s2.metric.tangent_space_dim

    # The tangent space likelihood
    tangent_space_likelihood = MultitaskGaussianLikelihood(num_tasks=tangent_space_dim,
                                                           noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
                                                           noise_prior=noise_prior)

    # Setting up a multitask kernel with a low-mean prior for
    # the lengthscale.
    kernel = MultitaskKernel(ScaleKernel(RBFKernel(lengthscale_prior=lengthscale_prior),  
                                         outputscale_prior=outputscale_prior), 
                                         num_tasks=tangent_space_dim,
                             rank=tangent_space_dim)

    if model_name == 'gplvm':
      # Creating the exact wrapped model
      wrapped_gplvm_on_r2_s2 = WrappedGPLVM(latent_dim, product_manifold_r2_s2, basepoint_function,
                                            toy_example_training_data, tangent_space_likelihood, tangent_kernel=kernel,
                                            trajectory_indexes=trajectory_indexes, initialization="pca")

    elif model_name == 'gpdm':
      # Creating the exact wrapped model with GPDM prior
      gpdm_kernel = ScaleKernel(RBFKernel())
      gpdm_likelihood = GaussianLikelihood()
      gpdm_trajectory_indexes = []
      gpdm_trajectory_indexes.append((0, int(trajectory_indexes[0].detach().numpy())))
      for i in range(trajectory_indexes.shape[0]-1):
          gpdm_trajectory_indexes.append((int(trajectory_indexes[i].detach().numpy()), int(trajectory_indexes[i+1].detach().numpy()) ))
      prior_x = GPDMPrior(gpdm_kernel, gpdm_likelihood, gpdm_trajectory_indexes, latent_dim)
      wrapped_gplvm_on_r2_s2 = WrappedGPLVM(latent_dim, product_manifold_r2_s2, basepoint_function,
                                            toy_example_training_data, tangent_space_likelihood, tangent_kernel=kernel,
                                            trajectory_indexes=trajectory_indexes, initialization="pca", 
                                            prior_x=prior_x)

    elif model_name == 'bcgplvm':
      # Creating the exact wrapped model with back constraints
      k_fct_euclidean = gpytorch.kernels.RBFKernel(active_dims=torch.tensor(list(range(0, dim_euclidean))))
      k_fct_sphere = SphereRiemannianGaussianKernel(dim_sphere,
                                                    active_dims=torch.tensor(list(range(dim_euclidean,
                                                                                        dim_euclidean +
                                                                                        dim_sphere + 1))))
      targets_kernel = k_fct_euclidean * k_fct_sphere
      targets_kernel.kernels[0].lengthscale = 0.2
      targets_kernel.kernels[1].lengthscale = 0.7
      wrapped_gplvm_on_r2_s2 = BackConstrainedWrappedGPLVM(latent_dim, product_manifold_r2_s2, basepoint_function,
                                                          toy_example_training_data, tangent_space_likelihood,
                                                          tangent_kernel=kernel, trajectory_indexes=trajectory_indexes,
                                                          initialization="pca",
                                                          targets_kernel=targets_kernel)
    elif model_name == 'bcgpdm':
      # Creating the exact Euclidean model with GPDM prior and back constraints
      # GPDM prior
      gpdm_kernel = ScaleKernel(RBFKernel())
      gpdm_likelihood = GaussianLikelihood()
      gpdm_trajectory_indexes = []
      gpdm_trajectory_indexes.append((0, int(trajectory_indexes[0].detach().numpy())))
      for i in range(trajectory_indexes.shape[0]-1):
          gpdm_trajectory_indexes.append((int(trajectory_indexes[i].detach().numpy()), int(trajectory_indexes[i+1].detach().numpy()) ))
      prior_x = GPDMPrior(gpdm_kernel, gpdm_likelihood, gpdm_trajectory_indexes, latent_dim)
      # Back constraints
      k_fct_euclidean = gpytorch.kernels.RBFKernel(active_dims=torch.tensor(list(range(0, dim_euclidean))))
      k_fct_sphere = SphereRiemannianGaussianKernel(dim_sphere,
                                                    active_dims=torch.tensor(list(range(dim_euclidean,
                                                                                        dim_euclidean +
                                                                                        dim_sphere + 1))))
      targets_kernel = k_fct_euclidean * k_fct_sphere
      targets_kernel.kernels[0].lengthscale = 0.2
      targets_kernel.kernels[1].lengthscale = 0.7
      # Model
      wrapped_gplvm_on_r2_s2 = BackConstrainedWrappedGPLVM(latent_dim, product_manifold_r2_s2, basepoint_function,
                                                          toy_example_training_data, tangent_space_likelihood,
                                                          tangent_kernel=kernel, trajectory_indexes=trajectory_indexes,
                                                          initialization="pca",
                                                          targets_kernel=targets_kernel,
                                                          prior_x=prior_x)
    else:
       raise NotImplementedError('Unknown model name.')

    return wrapped_gplvm_on_r2_s2, toy_example_training_data, toy_example_test_data, trajectory_indexes


def toy_experiment_on_letter_manifolds():
    """
    The main function. Here we train and visualize a WrappedGPLVM on
    the product of letter manifolds on R2 x S2.
    """
    # Training hyperparameters:
    model_name = 'gpdm'
    exp_name = "wrapped_" + model_name +  "_on_toy_experiment"
    max_iterations = 1000
    if model_name == 'bcgpdm':
      learning_rate = 0.01
    else:
       learning_rate = 0.025
    load_saved = True
    refine_discrete_geodesic = True

    # Visualization hyperparameters:
    uncertainty_and_volume_grid_size = 50
    discretized_grid_size = 50
    if model_name == 'gpdm':
        max_volume = 100
    else:
        max_volume = None

    # Load the model specified above.
    wrapped_gplvm_on_r2_s2, train_data, test_data, data_indexes = load_wgplvm_on_toy_experiment(model_name)

    # Train the model.
    wrapped_gplvm_on_r2_s2 = train_wrapped_gplvm(wrapped_gplvm_on_r2_s2, verbose=True, save_as=exp_name,
                                                 max_iterations=max_iterations, learning_rate=learning_rate,
                                                 load_saved=load_saved)

    # ------------------------ Visualizing ---------------------------
    wrapped_gplvm_on_r2_s2.eval()

    # Print some of the model parameters
    print(f'Lengthscale value value = {wrapped_gplvm_on_r2_s2.tangent_kernel.data_covar_module.base_kernel.lengthscale}')
    print(f'Outputscale value value = {wrapped_gplvm_on_r2_s2.tangent_kernel.data_covar_module.outputscale}')
    print(f'Latent variables = {wrapped_gplvm_on_r2_s2.latent_variable()}')

    # Check training error
    posterior = wrapped_gplvm_on_r2_s2(wrapped_gplvm_on_r2_s2.latent_variable())
    error = (posterior.mean - wrapped_gplvm_on_r2_s2.training_targets)
    print("Mean absolute error: ", torch.mean(torch.abs(error)))

    # Create plots folder
    plots_path = "/plots/"
    if not os.path.exists(ROOT_DIR.as_posix() + plots_path):
        print("Creating new plots folder for the experiment...")
        os.makedirs(ROOT_DIR.as_posix() + plots_path)

    # Creating the figure and axes
    fig1 = plt.figure(figsize=(7, 7))
    fig1_ax = fig1.gca()
    # Just plot the learned embeddings
    print("Simple visualization of learned embeddings.")
    visualize_latent_space(fig1_ax, wrapped_gplvm_on_r2_s2, grid_size_in_latent_space=uncertainty_and_volume_grid_size,
                           plot_uncertainty=False)
    fig1_ax.axis("off")
    fig1.tight_layout()
    fig1.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_LatentSpace.png")

    # Latent space, latent test data (BC) and GP uncertainty
    print("Visualizing latent space (with GP uncertainty)")
    visualize_latent_space(fig1_ax, wrapped_gplvm_on_r2_s2, grid_size_in_latent_space=uncertainty_and_volume_grid_size)
    fig1_ax.axis("off")
    fig1.tight_layout()

    # Visualizing latent space with volume
    fig2 = plt.figure(figsize=(7, 7))
    fig2_ax = fig2.gca()
    limits = visualize_latent_space(fig2_ax, wrapped_gplvm_on_r2_s2, plot_uncertainty=False, plot_volume=True,
                                    grid_size_in_latent_space=uncertainty_and_volume_grid_size, return_limits=True,
                                    max_volume=max_volume)
    fig2_ax.axis("off")
    fig2.tight_layout()

    # Compute geodesics
    p0 = wrapped_gplvm_on_r2_s2.latent_variable()[0]
    p1 = wrapped_gplvm_on_r2_s2.latent_variable()[-1]
    print("Computing discrete geodesics")
    discretized_path = ROOT_DIR.as_posix() + "/trained_models/" + exp_name + "_discretized.pt"
    wgpvlm_as_manifold = DiscreteGPLVMManifoldWrapper(wrapped_gplvm_on_r2_s2,
                                                      [torch.linspace(*limits[0], discretized_grid_size),
                                                        torch.linspace(*limits[1], discretized_grid_size)])
    
    if os.path.isfile(discretized_path):
        print("Pre-computed grid for trained model found.")
        wgpvlm_as_manifold = wgpvlm_as_manifold.from_path(wgpvlm_as_manifold, discretized_path)
    else:
      print("Pre-computed grid for trained model NOT found. Proceeding to compute it. This can take a while.")
      wgpvlm_as_manifold.fit()
      wgpvlm_as_manifold.save_discretized_manifold(discretized_path)
    geodesic, _ = wgpvlm_as_manifold.connecting_geodesic(p0, p1)
    if refine_discrete_geodesic:
        print("Refining discrete geodesics as continuous geodesics")
        wgpvlm_as_manifold = GPLVMManifoldWrapper(wrapped_gplvm_on_r2_s2)
        geodesic, _ = geodesic_minimizing_energy(wgpvlm_as_manifold,
                                                 p0, p1, 
                                                 init_curve=geodesic,
                                                 max_iter=100, lr=0.005)
    
    time = torch.linspace(0, 1, 50)
    # Visualize pullback geodesic in latent space.
    geodesic_points = geodesic(time).detach().numpy()
    fig1_ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], "-", color='orange', linewidth=5)
    fig2_ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], "-", color='orange', linewidth=5)
    # Compute and visualize Euclidean geodesics
    linear = torch.lerp(p0, p1, time[:, None])
    linear_points =linear.detach().numpy()
    fig1_ax.plot(linear_points[:, 0], linear_points[:, 1], "-", color='crimson', linewidth=5)
    fig2_ax.plot(linear_points[:, 0], linear_points[:, 1], "-", color='crimson',linewidth=5)
    # Save latent space plots
    fig1.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_LatentSpaceUncertainty_b.png")
    fig2.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_LatentSpaceMetricVol_b.png")

    # Plot the decoded predictions and geodesics on the ambient manifolds
    print("Plotting the manifold predictions")
    figr2 = plt.figure(figsize=[7, 7])  # Euclidean plot R2
    fig_ax_r2 = figr2.gca()

    # Sphere plot S2 with mayavi
    figs2 = mlab.figure(bgcolor=(1.0, 1.0, 1.0))
    visualize_wrapped_gplvm_on_r2_s2_mayavi(axes_r2=fig_ax_r2, fig_s2=figs2, wgplvm=wrapped_gplvm_on_r2_s2,
                                            traj_index=data_indexes, geodesic=geodesic(time), line=linear, 
                                            plot_reconstructions=False)

    fig_ax_r2.axis("off")
    # fig_ax_s2.axis("off")  # Uncomment if using matplotlib
    figr2.tight_layout()
    # figs2.tight_layout()  # Uncomment if using matplotlib
    figr2.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_R2_b.png")  # Uncomment to save
    # figs2.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_S2.png") # Uncomment if using matplotlib

    # Sphere plot S2 with mayavi (uncomment lines below if using Mayavi)
    # figs2.scene.camera.zoom(1.5)
    mlab.view(azimuth=0, elevation=90)
    mlab.savefig(ROOT_DIR.as_posix() + plots_path + f"{exp_name}_S2_b.png", magnification=2)  # Uncomment to save

    # Decode geodesics
    time2 = torch.linspace(0, 1, 200)
    geodesic_points2 = geodesic(time2)
    linear_points2 = torch.lerp(p0, p1, time[:, None])
    wgd_of_geodesic = wrapped_gplvm_on_r2_s2(geodesic_points2).mean.detach().numpy()
    wgd_of_line = wrapped_gplvm_on_r2_s2(linear_points2).mean.detach().numpy()

    # Compute percentage of points on sphere  
    g_on_sphere = np.isclose(np.linalg.norm(wgd_of_geodesic[:, 2:], axis=1), 1.0)
    print('Percentage of geodesic on sphere:', g_on_sphere.sum() / wgd_of_geodesic.shape[0])
    l_on_sphere = np.isclose(np.linalg.norm(wgd_of_line[:, 2:], axis=1), 1.0)
    print('Percentage of line on sphere:', l_on_sphere.sum() / wgd_of_line.shape[0])
    
    # Compute DTWD between demonstrations and geodesic/linear interpolation
    demos = train_data.tensor_split(data_indexes[:-1:].type(torch.int64))
    demos = [d.detach().numpy() for d in demos]
    dtwd_geodesic = np.array(get_dynamic_time_warping_distance_list(demos, [wgd_of_geodesic]))
    dtwd_line = np.array(get_dynamic_time_warping_distance_list(demos, [wgd_of_line]))
    print(f'DTWD geodesic: {np.mean(dtwd_geodesic):.2f} pm {np.std(dtwd_geodesic):.2f}')
    print(f'DTWD line: {np.mean(dtwd_line):.2f} pm {np.std(dtwd_line):.2f}')

    # Show plots
    mlab.show()
    plt.show()
    plt.close()


if __name__ == "__main__":
    toy_experiment_on_letter_manifolds()
