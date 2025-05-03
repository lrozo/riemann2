"""
Wrapped Gaussian Processes, first proposed in [Mallasto18], are a probabilistic
regression algorithm that works on Riemannian Manifolds. The core idea is
maintaining a Euclidean Gaussian Process in the tangent space, and use the
manifold's methods to move everything to the surface itself.

This implementation uses GPytorch for training/maintaining the Euclidean
Gaussian Process, and uses geomstats as a backend for all the geometrical
computations.

During training, the WrappedGaussianProcess object returns the tangent
space Normal distirbutions. On evaluation mode, the object returns a
WrappedGaussianDistribution, which is a wrapper around the tangent normal,
sending the mean and samples to the manifold using the exponential map.

[Mallasto18]: Wrapped Gaussian Process Regression on Riemannian Manifolds.
"""

from typing import Callable, Union, Tuple
from xml.etree.ElementInclude import include
import torch
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import RBFKernel
from gpytorch import add_jitter

from geomstats.geometry.manifold import Manifold
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.euclidean import EuclideanMetric

from .wrapped_gaussian_distribution import WrappedGaussianDistribution

from metrics.spd_affine_invariant_vectorized import SPDMetricAffineVectorized

from linear_operator.operators import MatmulLinearOperator
from linear_operator import to_dense
from functools import lru_cache
from gpytorch.distributions import MultivariateNormal

class WrappedGaussianProcess(gpytorch.models.ExactGP):
    """
    A class that implements a Wrapped Gaussian Process.

    Wrapped Gaussian Processes are a probabilistic method that allows
    for regression on Riemannian manifolds. For more details, check
    [Mallasto18].

    [Mallasto18]: Wrapped Gaussian Process Regression on Riemannian Manifolds.

    Attributes
    ----------
    - manifold (type: geomstats.geometry.manifold.Manifold or
      geomstats.geometry.manifold.RiemmanianMetric): an object
      that implements the exponential and logarithm maps for a
      given manifold. We usually use geomstats' objects (or
      wrappers thereof).

    - metric (type: a subclass of geomstat's RiemannianMetric)
      a metric object that implements exp, log and other geometric
      tools.

    - basepoint_function (type: a function that takes Tensors
      and returns Tensors): the "prior" of the Wrapped Gaussian
      Process; a function that returns, for each $x$, the point m(x)
      in the manifold from which the logarithms and exponentials are computed.

    - training_inputs (type: torch.Tensor): real-number tensor inputs.

    - training_targets (type: torch.Tensor): points in the manifold that are to
      be regressed.

    - tangent_space_likelihood (type: gpytorch.likelihoods.MultitaskGaussianLikelihood):
      the likelihood of the Euclidean GP in tangent spaces.

    - tangent_kernel (type: gpytorch.kernels.Kernel, optional): The kernel of the Euclidean
      GP. By default, it is a MultitaskKernel(ScaleKernel(RBFKernel)).

    - tangent_mean (type: gpytorch.priors.Prior, optional): The prior of
      the Euclidean GP in tangent space. By default, it is a ZeroPrior.

    - jitter_jacobian_cov_over_rows (type: float, optional): 
      “jitter” added to the diagonal of the covariance over the rows of the Jacobian.
      It ensures that the matrix that should be positive definite is positive definite.

    Methods
    -------
    - get_tangent_targets(train_inputs, train_targets): Computes the
      tangent vectors of the train_targets, using self.basepoint_function
      to compute the basepoints from which to compute the logarithm.

    - forward(x): Computes the tangent space distribution given by
      the tangent_prior and tangent_kernel, which is used for training
      the Euclidean GP.

    - tangent_space_distribution(x): computes the tangent space posterior
      distribution, used at evaluation mode.

    - __call__(*args, **kwargs): During training mode, returns the tangent space
      distribution as in forward. During evaluation mode, returns the posterior
      distribution as a WrappedGaussianDistribution which takes values on the
      manifold when sampling/computing means.

    - predict_jacobian(self, new_input): Predicts the transpose of
      the Jacobian at the given input. It returns a list of tensors containing
      the mean, covariance over rows (data posterior) and covariance over
      columns (task kernel).

    - predict_pullback_metric(self, new_inputs): Returns a point estimate of
      the pullback metric: the mean of J_y^T @ G @ J_y using the already computed
      means of the Jacobian transpose in the self.predict_jacobian method.
    """

    def __init__(
        self,
        manifold: Union[Manifold, RiemannianMetric],
        basepoint_function: Callable[[torch.Tensor], torch.Tensor],
        training_inputs: torch.Tensor,
        training_targets: torch.Tensor,
        tangent_space_likelihood: MultitaskGaussianLikelihood,
        tangent_kernel: gpytorch.kernels.Kernel = None,
        tangent_mean: gpytorch.priors.Prior = None,
        batch_independent: bool = False,
        rank: int = None,
        centering: bool = True,
        normalizing: bool = False,
        kernel_lengthscale_prior: gpytorch.priors.Prior = None,
        kernel_outputscale_prior: gpytorch.priors.Prior = None,
        jitter_jacobian_cov_over_rows: float = 1e-8,
    ):
        """
        Class constructor for WrappedGaussianProcesses.

        Parameters
        ----------
        - manifold (type: geomstats.geometry.manifold.Manifold or
        geomstats.geometry.manifold.RiemmanianMetric): an object
        that implements the exponential and logarithm maps for a
        given manifold. We usually use geomstats' objects (or
        wrappers thereof).

        - basepoint_function (type: a function that takes Tensors
        and returns Tensors): the "prior" of the Wrapped Gaussian
        Process; a function that returns, for each $x$, the point m(x)
        in the manifold from which the logarithms and exponentials are computed.

        - training_inputs (type: torch.Tensor): real-number tensor inputs.

        - training_targets (type: torch.Tensor): points in the manifold that are to
        be regressed.

        - tangent_space_likelihood (type: gpytorch.likelihoods.MultitaskGaussianLikelihood):
        the likelihood of the Euclidean GP in tangent spaces.

        - tangent_kernel (type: gpytorch.kernels.Kernel, optional): The kernel of the Euclidean
        GP. By default, it is a MultitaskKernel(ScaleKernel(RBFKernel)).

        - tangent_mean (type: gpytorch.priors.Prior, optional): The prior of
        the Euclidean GP in tangent space. By default, it is a ZeroPrior.

        - centering (type:bool, optimal): If True, we center the tangent targets to 0
          for training. The predictions are centered back around the original mean of
          the training tangent targets. 
        
        - normalizing (type:bool, optimal): If True, we normalize the tangent targets 
          for training. The predictions are unnormalized back to the original mean and
          standard deviation of the training tangent targets. 

        - kernel_lengthscale_prior (type: gpytorch.priors.Prior, optional):
          The prior on the lengthscale parameter of the default kernel.
          This is not used if a tangent_kernel is already passed as argument.

        - kernel_outputscale_prior (type: gpytorch.priors.Prior, optional):
          The prior on the scale parameter of the default kernel.
          This is not used if a tangent_kernel is already passed as argument.

        - jitter_jacobian_cov_over_rows (type: float, optional): 
          “jitter” that will be added to the diagonal of the covariance 
          over the rows of the Jacobian to ensure that it is positive definite.

        """
        if not isinstance(manifold, (Manifold, RiemannianMetric)):
            raise ValueError(
                "The given manifold should be of type Manifold or RiemannianMetric."
            )

        # A flag used for determining what to return in the forward:
        # either a multitask multivariate normal, or a batch-independent
        # multitask mutivariate normal.
        self.batch_independent = batch_independent

        # Manifold
        self.manifold = manifold

        # Defining the metric
        if isinstance(manifold, RiemannianMetric):
            self.metric = manifold
        else:
            self.metric = manifold.metric

        self.basepoint_function = basepoint_function

        # Move the targets to the tangent space.
        tangent_targets = self.get_tangent_targets(training_inputs, training_targets)

        if len(tangent_targets.shape) != 2:
            raise ValueError(
                "The given tangent targets should be vectors (i.e. with ndim=2)"
            )
        if len(training_inputs.shape) == 1:
            input_dim = 1
        else:
            _, input_dim = training_inputs.shape
        _, tangent_dim = tangent_targets.shape

        # Centering the tangent targets
        self.centering = centering
        self.normalizing = normalizing
        if centering or normalizing:
            print("Centering data...")
            self.tangent_data_mean = torch.mean(tangent_targets, 0)
        else:
            self.tangent_data_mean = torch.zeros(tangent_dim, dtype=tangent_targets.dtype)
        tangent_targets = tangent_targets - self.tangent_data_mean

        if normalizing:
            print("Normalizing data...")
            self.tangent_data_std = torch.std(tangent_targets, 0)
        else:
            self.tangent_data_std = torch.ones(tangent_dim, dtype=tangent_targets.dtype)
        tangent_targets = tangent_targets / self.tangent_data_std

        # Set up the GP in the tangent space.
        super().__init__(training_inputs, tangent_targets, tangent_space_likelihood)

        # Saving the training inputs and targets
        self.training_inputs = training_inputs
        self.training_targets = training_targets
        self.tangent_targets = tangent_targets

        # Set up the mean and kernel functions
        if tangent_mean is None:
            # Default tangent mean: zero.
            self.tangent_mean = gpytorch.means.MultitaskMean(
                gpytorch.means.ZeroMean(ard_num_dims=input_dim),
                num_tasks=tangent_dim,
            ).type(torch.float64)
        else:
            self.tangent_mean = tangent_mean

        if tangent_kernel is None:
            # Default tangent kernel: ScaleKernel(RBFKernel()).
            rank = rank or tangent_dim
            self.tangent_kernel = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(  # ard_num_dims=input_dim,  # Commented out to simplify training
                                               lengthscale_prior=kernel_lengthscale_prior), 
                    outputscale_prior=kernel_outputscale_prior
                ),
                rank=tangent_dim,
                num_tasks=tangent_dim,
            ).type(torch.float64)
        else:
            self.tangent_kernel = tangent_kernel

        self.tangent_space_likelihood = tangent_space_likelihood

        self.jitter_jacobian_cov_over_rows = jitter_jacobian_cov_over_rows

    def get_tangent_targets(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        basepoint_function: Callable = None,
        manifold: Union[Manifold, RiemannianMetric] = None,
    ) -> torch.Tensor:
        """
        Moves the train targets in the manifold to the tangent spaces
        based in self.basepoint_function(train_inputs).

        Parameters
        ----------
        - train_inputs (type: torch.Tensor)

        - train_targets (type: torch.Tensor): points in the manifold
          to which we will fit the regression.

        - basepoint_function (type: Callable[torch.Tensor -> torch.Tensor]):
          the basepoint function to be called. By default, we call the one
          stored in self. We added the possibility to provide one given that
          we need to be able to call this "statically" when initializing the
          latent variables of the Wrapped GPLVM using PCA.

        - manifold (type: Union[Manifold, RiemannianMetric]): the manifold in
          which the tangent space is computed. By default, we call the one
          stored in self. We added the possibility to provide one given that
          we need to be able to call this "statically" when initializing the
          latent variables of the Wrapped GPLVM using PCA.

        Returns
        -------
        - tangent_targets (type: torch.Tensor): the result of computing
          the logarithm to tra train_targets w.r.t the basepoints induced
          by the train_inputs.
        """
        if basepoint_function is None:
            basepoint_function = self.basepoint_function

        if manifold is None:
            manifold = self.manifold

        if isinstance(manifold, Manifold):
            metric = manifold.metric
        else:
            metric = manifold

        base_points = basepoint_function(train_inputs)
        tangent_targets = metric.log(train_targets, base_points).contiguous()

        if len(train_targets.shape) > 2:
            # Then we're dealing with matrices
            # TODO: with the new vectorized implementations, this
            # shouldn't happen.
            tangent_targets = tangent_targets.flatten(start_dim=1)

        return tangent_targets

    def forward(self, inps: torch.Tensor, **kwargs) -> MultitaskMultivariateNormal:
        """
        Returns the tangent space Gaussian distribution. This is standard
        practice when implementing GPs using GPyTorch.

        Parameters
        ----------
        - inps (type: torch.Tensor): inputs w.r.t which the mean and covariance
          are computed.

        - **kwargs: unused keyword arguments.

        Returns
        -------
        - tangent_space_distribution (type: gpytorch.distributions.MultitaskMultivariateNormal):
          the distribution in tangent space given by tangent_mean(inps)
          and tangent_kernel(inps)
        """
        mean_x = self.tangent_mean(inps)
        covar_x = self.tangent_kernel(inps)

        if self.batch_independent:
            dist_ = gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )
        else:
            dist_ = gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

        return dist_

    def tangent_space_distribution(
        self, X: torch.Tensor, **kwargs
    ) -> MultitaskMultivariateNormal:
        """
        Uses the inner Euclidean GP to return the MultitaskMultivariateNormal
        distribution in the tangent space.

        Parameters:
        ----------
        - X (torch.Tensor): Input values to the GP. Shape: (n_samples,) or (n_samples, n_features)

        - **kwargs: keyword arguments that would usually be passed to ExactGP.__call__.
        """
        assert not self.training

        # TODO: un-normalize (?).
        return super().__call__(X, **kwargs)

    def __call__(self, *args, **kwargs):
        # When optimizing, we train in the tangent space
        if self.training:
            return super().__call__(*args, **kwargs)

        # When predicting, we move from the tangent space to the manifold
        else:
            tangent_dist = super().__call__(*args, **kwargs)

            # The noise model (i.e. the likelihood) happens in
            # the tangent space, that's why we can't evaluate
            # something like self.likelihood(self(some_input)),
            # because self(some_input) returns a WrappedGaussianDistribution.
            with_likelihood = kwargs.get("with_likelihood", False)
            if with_likelihood:
                tangent_dist = self.tangent_space_likelihood(tangent_dist)

            X = args[0]
            basepoints = self.basepoint_function(X)

            # Center tangent data at original mean
            wgd_mean = tangent_dist.mean * self.tangent_data_std + self.tangent_data_mean

            # A Wrapped Gaussian Distirbution with
            # mean and samples in the manifold.
            wgd = WrappedGaussianDistribution(
                self.manifold,
                basepoints,
                wgd_mean,
                tangent_dist._covar,
            )
            return wgd

    def predict_jacobian(
        self, new_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicts the transpose of the Jacobian at a given
        new input. The distribution of the Jacobian is actually
        a Matrix normal, where

        - the mean is given by delta_K.T @ (Kx + (noise**2)I) @ Y,
          where delta_K is the Jacobian of the kernel w.r.t. the inputs
          x, Kx is the data kernel evaluated at the training inputs,
          and Y is the matrix of all tangent outputs.

        - the covariance among rows is given by the "posterior"
          w.r.t. the jacobian of the data kernel:
          delta_delta_K - delta_K.T @ (Kx + (noise**2)I) @ delta_K.

        - the covariance among columns is given by the task matrix
          (i.e. what the task kernel has learned the output dimensions'
          correlation should be).

        Currently, we only support an RBFKernel.

        For more context on how we compute the derivatives of a
        multitask kernel, we recommend checking the original paper where
        this kernel was proposed [Bonilla07], as well as Chap. 9.4 of [RW].
        We include these computations in the appendix of our paper. [TODO:
        add ref]

        TODO: Increase support to more than the RBFKernel: if we can
        compute the Jacobian of the kernel, we are set for more kernels.
        [Johnson20] is a good reference for computing these derivatives.

        References
        ----------

        [Bonilla07]: Multi-task Gaussian Process Prediction.
        [RW]: Gaussian Process for Machine Learning.
        [Johnson20]: Kernel Methods and their derivatives: Concept and
        perspectives for the Earth system sciences.

        Parameters
        ----------
        - new_inputs (torch.Tensor): a new input of the same size
          as the training inputs. (i.e. shape[-1] == latent dim)

        Returns
        -------
        - Mean of Jacobian transposed (torch.Tensor): the mean described above.
        - Covariance among rows (torch.Tensor): posterior coviarance among rows
        - Covariance among columns (torch.Tensor): task kernel, parametrizing
          the covariance among columns of the transpose Jacobian.

        """
        # Check that the shapes match
        n_points, _ = self.tangent_targets.shape

        # After this, new inputs has a batch dim.
        if len(new_input.shape) == 1:
            batched = False
            new_input = new_input.unsqueeze(0)
        elif len(new_input.shape) == 2:
            batched = True
        else:
            raise ValueError(
                "We were expecting either (b,) or (b,n) for the input shape, "
                "are you sure you are using Euclidean latent spaces?"
            )

        inputs = self.train_inputs[0]

        n_latent_dims = inputs.shape[1]
        if new_input.shape[1] != n_latent_dims:
            raise ValueError(
                "The input's shape didn't match the latent dimension of the "
                "stored training inputs "
                f"({new_input.shape[1]} vs. latent_dim={n_latent_dims})"
            )

        # Check that the data kernel is supported
        kernel_for_data = self.tangent_kernel.data_covar_module
        kernel_for_tasks = self.tangent_kernel.task_covar_module

        if not isinstance(self.tangent_kernel.data_covar_module.base_kernel, RBFKernel):
            raise ValueError(
                "We currently only support ScaleKernel(RBFKernel()) for the "
                "tangent kernel."
            )

        # Compute the kernel matrices and the lengthscale
        gram_matrix_tasks = kernel_for_tasks.covar_matrix
        gram_matrix_data = kernel_for_data(inputs) + (
            self.likelihood.noise**2
        ) * torch.eye(n_points)
        if kernel_for_data.base_kernel.lengthscale.shape[1] == 1:
            lengthscales = torch.Tensor(
                [kernel_for_data.base_kernel.lengthscale.item()] * n_latent_dims
            )
        else:
            lengthscales = kernel_for_data.base_kernel.lengthscale[0]

        # Compute the derivatives of the kernel
        # These commented lines are a non-vectorized version
        # of what the following lines do: computing the derivative
        # of the RBF kernel.

        # Non-vectorized version.
        # deltas_K_x = []
        # deltas_deltas_K_x = []
        # for x in new_input:
        #     # Computing the first derivative of the kernel.
        #     # This results, for each x in new inputs, in a
        #     # (n_points)x(n_latent_dims) matrix with the
        #     # derivative of k(x_n, x) w.r.t. the component r
        #     # of the inputs. deltas_K_x is, then, a batch of
        #     # matrices (b)x(n_points)x(n_latent_dims)
        #     delta_K_x = []
        #     for n in range(n_points):
        #         row_ = []
        #         for r in range(n_latent_dims):
        #             row_.append(
        #                 (lengthscales[r] ** (-2))
        #                 * (inputs[n, r] - x[r])
        #                 * kernel_for_data(inputs[n].unsqueeze(0), x.unsqueeze(0))
        #                 .evaluate()
        #                 .item()
        #             )
        #         delta_K_x.append(row_)
        #     deltas_K_x.append(delta_K_x)

        #     # For the RBF Kernel, the only thing that survives the double
        #     # derivation is the diagonal. deltas_deltas_K_x is a batch of
        #     # matrix of shape (b)x(n_latent_dims)x(n_latent_dims)
        #     deltas_deltas_K_x.append(
        #         torch.diag(
        #             torch.Tensor(
        #                 [
        #                     -(lengthscales[r] ** -2)
        #                     * kernel_for_data(x.unsqueeze(0), x.unsqueeze(0))
        #                     .evaluate()
        #                     .item()
        #                     for r in range(n_latent_dims)
        #                 ]
        #             )
        #         )
        #     )

        # deltas_K_x = torch.Tensor(deltas_K_x)
        # deltas_deltas_K_x = torch.cat([ddKx.unsqueeze(0) for ddKx in deltas_deltas_K_x])

        # Vectorized version:
        # Computing all the differences between inputs and new inputs
        # (an equivalent of inputs[n, r] - x[r] in the non-vectorized version)
        all_diffs = (lengthscales.unsqueeze(-2) ** -2) * (inputs.unsqueeze(0) - new_input.unsqueeze(1))

        # Evaluating the kernel on inputs and new inputs
        # (an equivalent of kernel_for_data(one_input, one_new_input) in
        # the non-vectorized version)
        mixed_kernel_evaluations = kernel_for_data(inputs, new_input).evaluate()

        # Multiplying them together
        deltas_K_x = mixed_kernel_evaluations.T.unsqueeze(-1) * all_diffs

        # A vector with k(x*, x*) for each x* in new_input.
        kernel_evaluations_on_new_input = kernel_for_data(new_input, new_input, diag=True)

        # A diagonal matrix with the inverse lengthscales
        # diagonal_lengthscales = torch.diag(-(lengthscales**-2))
        diagonal_lengthscales = torch.diag((lengthscales ** -2))

        # The product of them both, resulting in a diagonal matrix
        # for each new input.
        deltas_deltas_K_x = diagonal_lengthscales.unsqueeze(0) * kernel_evaluations_on_new_input.view(-1, 1, 1)

        # Computing the posterior mean and covariance.
        # gram_matrix_data_inverse = torch.linalg.inv(gram_matrix_data.evaluate())
        # Higher-precision inverse
        gram_matrix_data_inverse = self.get_gram_matrix_inverse(gram_matrix_data)

        left_half_of_product = torch.bmm(deltas_K_x.permute(0, 2, 1),
                                         gram_matrix_data_inverse.expand(len(deltas_K_x),
                                                                         *gram_matrix_data_inverse.shape),)
        jacobian_mean = torch.bmm(left_half_of_product, self.train_targets.unsqueeze(0).repeat(len(deltas_K_x), 1, 1),)

        jacobian_covariance_over_rows = deltas_deltas_K_x - torch.bmm(left_half_of_product, deltas_K_x)

        # Symmetrize for stability.
        # (otherwise some of the eigvals go complex)
        jacobian_covariance_over_rows = (jacobian_covariance_over_rows +
                                         jacobian_covariance_over_rows.permute(0, 2, 1)) / 2
        
        # Add “jitter” to the diagonal of the covariance 
        # This ensures that a matrix that should be positive definite is positive definite.
        # GPyTorch does the same with 1e-4 for uncertainty matrices
        # (see gpytorch.distributions.multivatiate_normal.py)
        jacobian_covariance_over_rows = add_jitter(jacobian_covariance_over_rows, 
                                                   self.jitter_jacobian_cov_over_rows)

        # Assert that the covariance over rows is SPD.
        # for j in -jacobian_covariance_over_rows:
        for j in jacobian_covariance_over_rows:
            if not torch.logical_and(torch.linalg.eigvals(j).imag == 0, torch.linalg.eigvals(j).real > 0.0).all():
                print(f"Negative or complex eigenvalues {torch.linalg.eigvals(j)} "
                    "on the covariance over rows with jitter", self.jitter_jacobian_cov_over_rows, 
                    ", weird!")

        jacobian_covariance_over_columns = (gram_matrix_tasks.evaluate().unsqueeze(0).repeat(len(deltas_K_x), 1, 1))

        if not batched:
            jacobian_mean = jacobian_mean.squeeze(0)
            jacobian_covariance_over_rows = jacobian_covariance_over_rows.squeeze(0)
            jacobian_covariance_over_columns = jacobian_covariance_over_columns.squeeze(0)

        # TODO: why do I have to consider -jacobian covariance over rows? The
        # posterior over the data is consistently giving me a SND instead of a SPD matrix.
        # **I need help with this**. Maybe I'm making a silly mistake with computing
        # the derivative of the kernel, but I've checked it a couple of times and
        # it seems to be implementing the formula correctly.
        # return (jacobian_mean, -jacobian_covariance_over_rows, jacobian_covariance_over_columns,)
        return jacobian_mean, jacobian_covariance_over_rows, jacobian_covariance_over_columns

    def predict_pullback_metric(
        self,
        new_inputs: torch.Tensor,
        include_jacobian_of_exp: bool = True,
        scale_for_covariance_term: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns the expected pullback metric according to the distribution
        of the Jacobian. Since we are able to predict the distribution of
        the Jacobian as a matrix-valued Normal, we can compute the expectation
        of the pullback J_y^T G J_y as

        E[J_y^T @ G @J_y] = E[J_y^T] @ G @ E[J_y] + S(J_y) Tr(G @ KTasks)

        where E[J_y^T] is the mean transpose Jacobian of the (Euclidean) map
        between the latent space and the tangent space/bundle, S(J_y) is the
        covariance among rows of J_y^T, and KTasks is the covariance among columns
        of J_y^T. These values are computed using the self.predict_jacobian method.

        If include_jacobian_of_exp is True, the computation includes an exterior term
        J_exp^T that multiplies both sides of the expectation. J_exp(y(x)) is a
        stochastic quantity, and we compute its average using autodifferentiation and
        ancestral sampling from the Euclidean GP on the tangent space/bundle.
        """
        # Check if we are batched or not.
        # (The proper checks are already inside self.predict_jacobian,
        # but we need to know if we are batched or not, and we need
        # the extra dim for the matrix multiplications.)
        if len(new_inputs.shape) == 1:
            batched = False
            new_inputs = new_inputs.unsqueeze(0)
        else:
            batched = True

        # Predict the transposed Jacobian's mean and covariances.
        (
            jacobian_mean,
            jacobian_cov_over_rows,
            jacobian_cov_over_cols,
        ) = self.predict_jacobian(new_inputs)

        # The metric is also stochastic, but we settled for dealing with the "mean"
        # defined as taking the mean of the predictions.
        if isinstance(self.metric, (EuclideanMetric, SPDMetricAffineVectorized)):
            # In the Euclidean case (i.e. for Euclidean GPLVMs)
            # we won't compute the jacobian of exp. It is the identity.
            include_jacobian_of_exp = False

        if include_jacobian_of_exp:
            metric_matrix = (
                torch.eye(self.metric.ambient_space_dim)
                .unsqueeze(0)
                .repeat(len(new_inputs), 1, 1)
            )
        else:
            prediction_mean = self(new_inputs).mean
            metric_matrix = self.metric.metric_matrix(prediction_mean)

            if len(metric_matrix.shape) == 2:
                metric_matrix = metric_matrix.unsqueeze(0).repeat(
                    len(prediction_mean), 1, 1
                )

        # Computing J_exp if we are asked to
        if include_jacobian_of_exp:
            samples = self.tangent_space_distribution(new_inputs).mean
            basepoints = self.basepoint_function(new_inputs)
            jacobians = [
                torch.autograd.functional.jacobian(
                    self.metric.exp,
                    (tangent_vec, basepoint),
                )
                for tangent_vec, basepoint in zip(samples, basepoints)
            ]

            # The Jacobian is only w.r.t the tangent vectors.
            jacobian = torch.cat([j[0] for j in jacobians])

            # Updating the metric matrix to be J_exp^T @ G @ J_exp
            metric_matrix = torch.bmm(
                torch.bmm(jacobian.permute(0, 2, 1), metric_matrix), jacobian
            )

        # Compute the product E[J_y^T] @ G @ E[J_y]
        left_half_of_product = torch.bmm(jacobian_mean, metric_matrix)
        first_summand = torch.bmm(left_half_of_product, jacobian_mean.permute(0, 2, 1))

        # Compute S(J_y) Tr(G @ KTasks)
        traces = (
            torch.Tensor(
                [
                    torch.trace(matrix)
                    for matrix in torch.bmm(metric_matrix, jacobian_cov_over_cols)
                ]
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        second_summand = scale_for_covariance_term * traces * jacobian_cov_over_rows

        # Return the pullback
        pullback = first_summand + second_summand
        if not batched:
            pullback = pullback.squeeze(0)

        return pullback

    @lru_cache()
    def get_gram_matrix_inverse(self, gram_matrix_data):
        with gpytorch.settings.max_root_decomposition_size(3000):
            gram_matrix_data_inv_root = to_dense(gram_matrix_data.root_inv_decomposition().root)
        return MatmulLinearOperator(gram_matrix_data_inv_root, gram_matrix_data_inv_root.transpose(-1, -2)).to_dense()


