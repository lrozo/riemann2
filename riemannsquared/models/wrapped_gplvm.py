"""
Implements Wrapped Gaussian Process Latent Variable Models
using geomstats as the geometry backend, and gpytorch for
the Gaussian Process.

This relies on our implementation of Wrapped Gaussian Processes,
which can be found in the sibling file ./wrapped_gaussian_process.py

Wrapped GPLVMs were proposed in [Mallasto19], as a pretty straight-forward
follow-up on the original work on Wrapped GPs [Mallasto18]. The core difference
between GPs and GPLVMs is that the "inputs" (now called "latent variables")
are not provided during training, but rather are to be inferred, optimized
by maximizing a MLE/MAP estimate (or an ELBO estimate using variational 
approximations).

[Mallasto18]: Wrapped Gaussian Process Regression on Riemannian Manifolds.
[Mallasto19]: Probabilistic Riemannian submanifold learning with wrapped
              Gaussian process latent variable models.
"""
from typing import Union, Callable

import numpy as np

import torch
from torch.optim.adam import Adam

import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.models.gplvm import PointLatentVariable, MAPLatentVariable
from gpytorch.priors.prior import Prior
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel

from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric

from .wrapped_gaussian_process import WrappedGaussianProcess
from riemannsquared.initialization.gplvm_initialization import trajectories_stress_loss_initialization
from riemannsquared.models.latent_variable import BackConstraintsLatentVariable


class WrappedGPLVM(WrappedGaussianProcess):
    """
    Implements a Wrapped Gaussian Process Latent Variable Model.

    Wrapped GPLVMs learn a Euclidean latent space from data that lies
    on a manifold. In other words, it learns a submanifold that is
    constraint-aware. For more details, check [Mallasto19].

    [Mallasto19]: Probabilistic Riemannian submanifold learning with
                  wrapped Gaussian process latent variable models.

    New Attributes (besides those in WrappedGaussianProcess)
    --------------------------------------------------------

    - manifold_targets (torch.Tensor): the outputs of the GPLVM lying
      on the manifold.

    - latent_variable (gpytorch's PointLatentVariable): A latent variable
      that is optimized using MLE.

    New Methods
    -----------

    - sample_latent_variable (() -> Torch.Tensor): returns a sample of the
      latent variables. Since we are currently only supporting MLE estimates,
      it simply returns the latent variables themselves.

    """

    def __init__(
        self,
        latent_dim: int,
        manifold: Union[Manifold, RiemannianMetric],
        basepoint_function: Callable[[torch.Tensor], torch.Tensor],
        training_targets: torch.Tensor,
        tangent_space_likelihood: MultitaskGaussianLikelihood,
        tangent_kernel: gpytorch.kernels.Kernel = None,
        tangent_mean: gpytorch.priors.Prior = None,
        initial_latent_variables: torch.nn.Parameter = None,
        batch_independent: bool = False,
        initialization: str = "pca",
        trajectory_indexes: torch.Tensor = None,
        rank: int = None,
        prior_x = None,
        **kwargs
    ):
        """
        Constructs the Wrapped GPLVM.

        Parameters
        ----------
        - latent_dim (type: int): The dimension of the latent space.

        - manifold (type: Manifold or RiemannianMetric): the manifold in
          which the observed data lives. This is expected to be a geomstats object,
          or a class that implements methods like the exponential and logarithm
          maps.

        - basepoint_function (type: a function that takes Tensors
          and returns Tensors): the "prior" m(x) of the Wrapped Gaussian
          Process; a function that returns, for each $x$, a point m(x)
          in the manifold from which the logarithms and exponentials are computed.
          m(x) is then a function that "decides" which base point to use for
          a certain latent variable.

        - training_targets (type: torch.Tensor[float64]): the data, laying on
          the manifold.

        - tangent_space_likelihood (type: gpytorch.likelihoods.MultitaskGaussianLikelihood):
          the likelihood of the Euclidean GP in tangent spaces.

        - tangent_kernel (type: gpytorch.kernels.Kernel, optional): The kernel of the Euclidean
          GP. By default, it is a MultitaskKernel(ScaleKernel(RBFKernel)).

        - tangent_mean (type: gpytorch.priors.Prior, optional): The prior of
          the Euclidean GP in tangent space. By default, it is a ZeroPrior.

        - initial_latent_variables (type: torch.Tensor, optional): a custom
          initialization for the latent variables. If provided, it overrides
          the initialization kwarg below.

        - batch_independent (type: bool): determines whether we have a multitask (False)
          or batch independent (True) set-up for the tangent space GPLVM.

        - initialization (type: str): One of ["random", "pca"]. By default, we have
          pca initialization. However, if initial_latent_variables is provided, we
          will use those of course.
        """
        n_points, *_ = training_targets.shape

        # Initializing the latent variables.
        # TODO: add support for PCA initialization.
        if initial_latent_variables is None:
            # Default latent variables: random ones.
            if initialization == "random":
                initial_latent_variables = torch.nn.Parameter(
                    torch.randn(n_points, latent_dim, dtype=training_targets.dtype)
                )
            elif initialization == "pca":
                # The PCA initialization projects the data laying on the manifold.
                # This is because, to compute the tangent targets and to perform PCA
                # in the tangent space we would need to have the basepoints, and to
                # have the basepoints we would need a first choice of latent variables...
                # This is a chicken-or-egg problem, and so we settle for having a first
                # approximation by just projecting the manifold data.
                _, _, V = torch.pca_lowrank(
                    training_targets.flatten(start_dim=1), q=latent_dim
                )
                initial_latent_variables = torch.nn.Parameter(
                    torch.matmul(
                        training_targets.flatten(start_dim=1), V[:, :latent_dim]
                    )
                )
            elif initialization == "stress":
                # Stress initialization aims at having latent variables whose distance closely matches the distances of
                # observations in the ambient space.
                initial_latent_variables = trajectories_stress_loss_initialization(training_targets, latent_dim, trajectory_indexes, manifold.metric.squared_dist)
            else:
                raise ValueError(
                    f"Expected initalization to be in ['random', 'pca', 'stress], but got {initialization}"
                )

        # Defining an (MLE-estimated) latent variable.
        if prior_x is None: 
            latent_variable = PointLatentVariable(n_points, latent_dim, initial_latent_variables)
        else:
            latent_variable = MAPLatentVariable(n_points, latent_dim, initial_latent_variables, prior_x)

        # We will need access to the manifold targets.
        # These are actually still the same ones as in
        # self.targets. TODO: see if we can clean this
        # after merging with future branches.
        self.manifold_targets = training_targets

        # Initializing the internal Wrapped GP that
        # governs the relationship between the latent codes
        # and the data on the manifold.
        super().__init__(
            manifold,
            basepoint_function,
            latent_variable.X,
            training_targets,
            tangent_space_likelihood,
            tangent_kernel,
            tangent_mean,
            batch_independent=batch_independent,
            rank=rank,
            **kwargs
        )

        self.latent_variable = latent_variable

    def sample_latent_variable(self) -> torch.Tensor:
        """
        Returns a sample of the latent variable.

        Returns
        -------
        - sample (type: torch.Tensor of shape (n_points, latent_dim))
        """
        sample = self.latent_variable()
        return sample


class BackConstrainedWrappedGPLVM(WrappedGaussianProcess):
    """
    Implements a Back-Constrained Wrapped Gaussian Process Latent Variable Model.

    Wrapped GPLVMs learn a Euclidean latent space from data that lies
    on a manifold. In other words, it learns a submanifold that is
    constraint-aware. For more details, check [Mallasto19].

    [Mallasto19]: Probabilistic Riemannian submanifold learning with
                  wrapped Gaussian process latent variable models.

    New Attributes (besides those in WrappedGaussianProcess)
    --------------------------------------------------------

    - manifold_targets (torch.Tensor): the outputs of the GPLVM lying
      on the manifold.

    - latent_variable (BackConstraintsLatentVariable): A latent variable
      that is optimized using MLE.

    New Methods
    -----------

    - sample_latent_variable (() -> Torch.Tensor): returns a sample of the
      latent variables. Since we are currently only supporting MLE estimates,
      it simply returns the latent variables themselves.

    """

    def __init__(
        self,
        latent_dim: int,
        manifold: Union[Manifold, RiemannianMetric],
        basepoint_function: Callable[[torch.Tensor], torch.Tensor],
        training_targets: torch.Tensor,
        tangent_space_likelihood: MultitaskGaussianLikelihood,
        tangent_kernel: gpytorch.kernels.Kernel = None,
        tangent_mean: gpytorch.priors.Prior = None,
        initial_latent_variables: torch.nn.Parameter = None,
        batch_independent: bool = False,
        initialization: str = "pca",
        trajectory_indexes: torch.Tensor = None,
        rank: int = None,
        targets_kernel: Kernel = None,  # TODO add in doc
        weight_prior: Prior = None,  # TODO add in doc
        prior_x: Prior = None,  # TODO add in doc
        **kwargs
    ):
        """
        Constructs the Wrapped GPLVM.

        Parameters
        ----------
        - latent_dim (type: int): The dimension of the latent space.

        - manifold (type: Manifold or RiemannianMetric): the manifold in
          which the observed data lives. This is expected to be a geomstats object,
          or a class that implements methods like the exponential and logarithm
          maps.

        - basepoint_function (type: a function that takes Tensors
          and returns Tensors): the "prior" m(x) of the Wrapped Gaussian
          Process; a function that returns, for each $x$, a point m(x)
          in the manifold from which the logarithms and exponentials are computed.
          m(x) is then a function that "decides" which base point to use for
          a certain latent variable.

        - training_targets (type: torch.Tensor[float64]): the data, laying on
          the manifold.

        - tangent_space_likelihood (type: gpytorch.likelihoods.MultitaskGaussianLikelihood):
          the likelihood of the Euclidean GP in tangent spaces.

        - tangent_kernel (type: gpytorch.kernels.Kernel, optional): The kernel of the Euclidean
          GP. By default, it is a MultitaskKernel(ScaleKernel(RBFKernel)).

        - tangent_mean (type: gpytorch.priors.Prior, optional): The prior of
          the Euclidean GP in tangent space. By default, it is a ZeroPrior.

        - initial_latent_variables (type: torch.Tensor, optional): a custom
          initialization for the latent variables. If provided, it overrides
          the initialization kwarg below.

        - batch_independent (type: bool): determines whether we have a multitask (False)
          or batch independent (True) set-up for the tangent space GPLVM.

        - initialization (type: str): One of ["random", "pca"]. By default, we have
          pca initialization. However, if initial_latent_variables is provided, we
          will use those of course.
        """
        n_points, *_ = training_targets.shape

        # Initializing the latent variables.
        # TODO: add support for PCA initialization.
        if initial_latent_variables is None:
            # Default latent variables: random ones.
            if initialization == "random":
                initial_latent_variables = torch.nn.Parameter(
                    torch.randn(n_points, latent_dim, dtype=training_targets.dtype)
                )
            elif initialization == "pca":
                # The PCA initialization projects the data laying on the manifold.
                # This is because, to compute the tangent targets and to perform PCA
                # in the tangent space we would need to have the basepoints, and to
                # have the basepoints we would need a first choice of latent variables...
                # This is a chicken-or-egg problem, and so we settle for having a first
                # approximation by just projecting the manifold data.
                _, _, V = torch.pca_lowrank(
                    training_targets.flatten(start_dim=1), q=latent_dim
                )
                initial_latent_variables = torch.nn.Parameter(
                    torch.matmul(
                        training_targets.flatten(start_dim=1), V[:, :latent_dim]
                    )
                )
            elif initialization == "stress":
                # Stress initialization aims at having latent variables whose distance closely matches the distances of
                # observations in the ambient space.
                initial_latent_variables = trajectories_stress_loss_initialization(training_targets, latent_dim,
                                                                                   trajectory_indexes,
                                                                                   manifold.metric.squared_dist)
            else:
                raise ValueError(
                    f"Expected initalization to be in ['random', 'pca', 'stress], but got {initialization}"
                )

        # Defining an (MLE-estimated) latent variable.
        if targets_kernel is None:
            targets_kernel = ScaleKernel(RBFKernel())
        kernel_backconstraints = targets_kernel(training_targets).evaluate()

        # Solve least square problem
        regularization = 1e-6 * torch.eye(n_points)
        weight_init = torch.matmul(torch.inverse(kernel_backconstraints + regularization), initial_latent_variables)
        # weight_init = torch.linalg.lstsq(kernel_backconstraints, initial_latent_variables).solution

        # Solve least square problem
        # For numerical stability, we get several solutions and keep the one with smallest values
        # ls_sols = [torch.linalg.lstsq(kernel_backconstraints, initial_latent_variables).solution for i in range(0, 3)]
        # ls_norms = [torch.norm(ls_sol).item() for ls_sol in ls_sols]
        # weight_init = ls_sols[np.argmin(ls_norms)]

        latent_variable = BackConstraintsLatentVariable(n_points, latent_dim, training_targets, targets_kernel, 
                                                        weight_prior, weight_init, prior_x=prior_x)


        # We will need access to the manifold targets.
        # These are actually still the same ones as in
        # self.targets. TODO: see if we can clean this
        # after merging with future branches.
        self.manifold_targets = training_targets

        # Initializing the internal Wrapped GP that
        # governs the relationship between the latent codes
        # and the data on the manifold.
        super().__init__(
            manifold,
            basepoint_function,
            latent_variable(),
            training_targets,
            tangent_space_likelihood,
            tangent_kernel,
            tangent_mean,
            batch_independent=batch_independent,
            rank=rank,
            **kwargs
        )

        self.latent_variable = latent_variable

    @property
    def train_inputs(self) -> tuple[torch.Tensor]:
        # train_inputs follow the shape as in gpytorch.models.ExactGP __init__()
        train_inputs = self.latent_variable()
        if train_inputs is not None and torch.is_tensor(train_inputs):
            train_inputs = (train_inputs,)
        return tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs) if train_inputs is not None else None

    @train_inputs.setter
    def train_inputs(self, _) -> None:
        # cannot set train_inputs directly as they a defined by the LatentVariable object
        pass

    def sample_latent_variable(self) -> torch.Tensor:
        """
        Returns a sample of the latent variable.

        Returns
        -------
        - sample (type: torch.Tensor of shape (n_points, latent_dim))
        """
        sample = self.latent_variable()
        return sample
