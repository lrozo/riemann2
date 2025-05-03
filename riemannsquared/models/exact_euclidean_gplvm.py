"""
This implementation of an exact Euclidean GPLVM
uses our Wrapped GPLVM, but forces the manifold to
be Euclidean and the multitask kernel to have diagonal
task covariance.

Implementing it this way saves us from having to
re-implement pullback predictions.

"""

import torch

import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.priors.prior import Prior
from gpytorch.kernels import Kernel

from geomstats.geometry.euclidean import Euclidean

from metrics.euclidean import AugmentedCanonicalEuclideanMetric

from .wrapped_gplvm import WrappedGPLVM, BackConstrainedWrappedGPLVM

torch.set_default_dtype(torch.float64)


class ExactWGPLVM(WrappedGPLVM):
    """
    An implementation of an exact GPLVM on the Euclidean
    manifold of the dimension given by the training targets.

    In its current implementation, it is just a Wrapper around
    our WrappedGPLVM (hehe), creating internally an instance
    of an Euclidean manifold and a constant basepoint function
    on the origin.

    By default, it uses a multitask kernel with rank 0 (i.e.
    a diagonal one) and a zero mean.

    Attributes and methods can be found in the documentation
    for WrappedGPLVM.
    """

    def __init__(
        self,
        latent_dim: int,
        training_targets: torch.Tensor,
        likelihood: MultitaskGaussianLikelihood,
        trajectory_indexes: torch.Tensor = None,
        kernel: gpytorch.kernels.Kernel = None,
        mean: gpytorch.priors.Prior = None,
        initial_latent_variables: torch.nn.Parameter = None,
        batch_independent: bool = False,
        initialization: str = "pca",
        kernel_lengthscale_prior: gpytorch.priors.Prior = None,
        kernel_outputscale_prior: gpytorch.priors.Prior = None,
        prior_x = None,
        **kwargs
    ):
        """
        Constructs the ExactEuclideanGPLVM.

        Parameters
        ----------
        - latent_dim (type: int): The dimension of the latent space.

        - training_targets (type: torch.Tensor[float64]): the data we want to train
          on, assumed to be in some Euclidean space.

        - likelihood (type: gpytorch.likelihoods.MultitaskGaussianLikelihood):
          the likelihood of the GP in the whole ambient manifold.

        - kernel (type: gpytorch.kernels.Kernel, optional): The kernel of the GP.
          By default, it is a MultitaskKernel(ScaleKernel(RBFKernel)) with 0 rank
          (i.e. a diagonal task covariance).

        - mean (type: gpytorch.means.Mean, optional): The prior of the GP. By default,
          it is a ZeroMean.

        - initial_latent_variables (type: torch.Tensor, optional): a custom
          initialization for the latent variables. If provided, it overrides
          the initialization kwarg below.

        - batch_independent (type: bool, optional): determines whether we have a
          multitask (False) or batch independent (True) set-up for the GPLVM.
          By default, it is True

        - initialization (type: str): One of ["random", "pca"]. By default, we have
          pca initialization. However, if initial_latent_variables is provided, we
          will use those of course.

        - kernel_lengthscale_prior (type: gpytorch.priors.Prior, optional):
          The prior on the lengthscale parameter of the default kernel.
          This is not used if a kernel is already passed as argument.

        - kernel_outputscale_prior (type: gpytorch.priors.Prior, optional):
          The prior on the scale parameter of the default kernel.
          This is not used if a kernel is already passed as argument.
        """
        _, ambient_dim = training_targets.shape
        manifold = Euclidean(ambient_dim, equip=False)
        manifold.equip_with_metric(AugmentedCanonicalEuclideanMetric)

        basepoint_function = lambda _: torch.zeros((ambient_dim,))

        if kernel is None:
            if batch_independent:
                # Default kernel: ScaleKernel(RBFKernel()) w. batch independence.
                kernel = gpytorch.kernels.MultitaskKernel(
                    gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel(  # ard_num_dims=latent_dim,  # Commented out to simplify training
                                                   lengthscale_prior=kernel_lengthscale_prior), 
                    outputscale_prior=kernel_outputscale_prior
                    ),
                    num_tasks=ambient_dim,
                    rank=0,  # i.e. diagonal task covariance.
                )
            else:
                kernel = gpytorch.kernels.MultitaskKernel(
                    gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel(  # ard_num_dims=latent_dim,  # Commented out to simplify training
                                                   lengthscale_prior=kernel_lengthscale_prior), 
                        outputscale_prior=kernel_outputscale_prior
                    ),
                    num_tasks=ambient_dim,
                    rank=ambient_dim,
                )

        if mean is None:
            # Default tangent mean: Constant and B.Indep.
            mean = gpytorch.means.MultitaskMean(
                gpytorch.means.ZeroMean(ard_num_dims=latent_dim),
                num_tasks=ambient_dim,
            ).type(torch.float64)

        super().__init__(latent_dim, manifold, basepoint_function, training_targets, likelihood, kernel, mean,
                         initial_latent_variables, batch_independent,
                         initialization, trajectory_indexes, prior_x=prior_x, **kwargs)


class BackConstrainedExactWGPLVM(BackConstrainedWrappedGPLVM):
    """
    An implementation of an exact GPLVM on the Euclidean
    manifold of the dimension given by the training targets.

    In its current implementation, it is just a Wrapper around
    our WrappedGPLVM (hehe), creating internally an instance
    of an Euclidean manifold and a constant basepoint function
    on the origin.

    By default, it uses a multitask kernel with rank 0 (i.e.
    a diagonal one) and a zero mean.

    Attributes and methods can be found in the documentation
    for WrappedGPLVM.
    """

    def __init__(
        self,
        latent_dim: int,
        training_targets: torch.Tensor,
        likelihood: MultitaskGaussianLikelihood,
        trajectory_indexes: torch.Tensor,  # TODO add in doc
        kernel: gpytorch.kernels.Kernel = None,
        mean: gpytorch.priors.Prior = None,
        initial_latent_variables: torch.nn.Parameter = None,
        batch_independent: bool = False,
        initialization: str = "pca",
        kernel_lengthscale_prior: gpytorch.priors.Prior = None,
        kernel_outputscale_prior: gpytorch.priors.Prior = None,
        targets_kernel: Kernel = None,  # TODO add in doc
        weight_prior: Prior = None,  # TODO add in doc
        prior_x: Prior = None,  # TODO add in doc
        **kwargs
    ):
        """
        Constructs the ExactEuclideanGPLVM.

        Parameters
        ----------
        - latent_dim (type: int): The dimension of the latent space.

        - training_targets (type: torch.Tensor[float64]): the data we want to train
          on, assumed to be in some Euclidean space.

        - likelihood (type: gpytorch.likelihoods.MultitaskGaussianLikelihood):
          the likelihood of the GP in the whole ambient manifold.

        - kernel (type: gpytorch.kernels.Kernel, optional): The kernel of the GP.
          By default, it is a MultitaskKernel(ScaleKernel(RBFKernel)) with 0 rank
          (i.e. a diagonal task covariance).

        - mean (type: gpytorch.means.Mean, optional): The prior of the GP. By default,
          it is a ZeroMean.

        - initial_latent_variables (type: torch.Tensor, optional): a custom
          initialization for the latent variables. If provided, it overrides
          the initialization kwarg below.

        - batch_independent (type: bool, optional): determines whether we have a
          multitask (False) or batch independent (True) set-up for the GPLVM.
          By default, it is True

        - initialization (type: str): One of ["random", "pca"]. By default, we have
          pca initialization. However, if initial_latent_variables is provided, we
          will use those of course.

        - kernel_lengthscale_prior (type: gpytorch.priors.Prior, optional):
          The prior on the lengthscale parameter of the default kernel.
          This is not used if a tangent_kernel is already passed as argument.

        - kernel_outputscale_prior (type: gpytorch.priors.Prior, optional):
          The prior on the scale parameter of the default kernel.
          This is not used if a tangent_kernel is already passed as argument.

        - targets_kernel (type: gpytorch.kernels.Kernel, optional):
          The kernel defining the relationship between the observations for the back constraints.

        - weight_prior (type: gpytorch.priors.Prior, optional):
          The prior for the weight parameter of the back constraints.
        """
        _, ambient_dim = training_targets.shape
        manifold = Euclidean(ambient_dim, equip=False)
        manifold.equip_with_metric(AugmentedCanonicalEuclideanMetric)
        
        basepoint_function = lambda _: torch.zeros((1, ambient_dim))

        if kernel is None:
            if batch_independent:
                # Default kernel: ScaleKernel(RBFKernel()) w. batch independence.
                kernel = gpytorch.kernels.MultitaskKernel(
                    gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel(  # ard_num_dims=latent_dim,  # Commented out to simplify training
                                                   lengthscale_prior=kernel_lengthscale_prior), 
                    outputscale_prior=kernel_outputscale_prior
                    ),
                    num_tasks=ambient_dim,
                    rank=0,  # i.e. diagonal task covariance.
                )
            else:
                kernel = gpytorch.kernels.MultitaskKernel(
                    gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel(  # ard_num_dims=latent_dim,  # Commented out to simplify training
                                                   lengthscale_prior=kernel_lengthscale_prior), 
                    outputscale_prior=kernel_outputscale_prior
                    ),
                    num_tasks=ambient_dim,
                    rank=ambient_dim,
                )

        if mean is None:
            # Default tangent mean: Constant and B.Indep.
            mean = gpytorch.means.MultitaskMean(
                gpytorch.means.ZeroMean(ard_num_dims=latent_dim),
                num_tasks=ambient_dim,
            ).type(torch.float64)

        super().__init__(latent_dim, manifold, basepoint_function, training_targets, likelihood, kernel, mean,
                         initial_latent_variables, batch_independent, initialization, trajectory_indexes,
                         targets_kernel=targets_kernel, weight_prior=weight_prior, prior_x=prior_x, **kwargs)