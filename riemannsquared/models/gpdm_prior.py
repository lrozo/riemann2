import torch
import gpytorch
from torch.nn import Module as TModule
from gpytorch.distributions import MultivariateNormal


class GPDMPrior(gpytorch.priors.Prior):
    """
    gpdm prior which consists of a standard gaussian on x_1 and a "markov gaussian" where the kernel matrix
    is constructed of x_1,...x_N-1 to evalue the probability for x_2,...,x_N. 
    p(X) = p(x_1) * prod_d=1^Q N(X_2:N,d | 0, K_1:N_1 + noise_d,x) 
    """

    def __init__(self, kernel_module: gpytorch.kernels.Kernel, likelihood: gpytorch.likelihoods.Likelihood, trajectory_indices: list[tuple[int, int]], latent_dim: int, weighting_factor=1.0):
        """
        Parameters
        -
        normal_distribution: expects a class which inherits from MultivariateNormal depending on the manifold
            gpytorch.distributions.MultivariateNormal  if manifold is euclidean
            LorentzWrappedNormal if manifold is lorentz
        """
        TModule.__init__(self)
        self.kernel_module = kernel_module
        self.likelihood = likelihood
        self.trajectory_indices = trajectory_indices
        self.weighting_factor = weighting_factor

        mean, covariance_matrix = torch.zeros(latent_dim), torch.eye(latent_dim)
        self.first_point_distribution = MultivariateNormal(mean, covariance_matrix)

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters: 
        X: torch.Size([N, D_x])   N latent variables of dimension D_x

        Returns:
        log_prob: torch.Size([1]) scalar value
        """
        log_p_x1 = self.first_point_log_prob(X)
        X_lower, X_upper = self.get_lower_and_upper_latent_variables(X)

        gpdm_kernel = self.kernel_module(X_lower)
        gpdm_prior = MultivariateNormal(X_lower.T, gpdm_kernel)
        marginal_likelihood = self.likelihood(gpdm_prior)
        log_prob_X_upper = marginal_likelihood.log_prob(X_upper.T).sum()
        return (log_p_x1 + log_prob_X_upper) * self.weighting_factor

    def get_lower_and_upper_latent_variables(self, X: torch.Tensor, trajectory_indices: list[tuple[int, int]] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if trajectory_indices is None:
            trajectory_indices = self.trajectory_indices
        N = X.shape[0]
        start_indices = [start for start, _ in trajectory_indices]
        end_indices_inclusive = [end-1 for _, end in trajectory_indices]

        upper_mask = torch.ones(N, dtype=torch.bool)
        upper_mask[start_indices] = False
        X_upper = X[upper_mask]
        lower_mask = torch.ones(N, dtype=torch.bool)
        lower_mask[end_indices_inclusive] = False
        X_lower = X[lower_mask]
        return X_lower, X_upper

    def get_first_latent_variables(self, X: torch.Tensor, trajectory_indices: list[tuple[int, int]] = None) -> torch.Tensor:
        if trajectory_indices is None:
            trajectory_indices = self.trajectory_indices
        start_indices = [start for start, _ in trajectory_indices]
        return X[start_indices]

    def first_point_log_prob(self, X: torch.Tensor, trajectory_indices: list[tuple[int, int]] = None) -> torch.Tensor:
        X_1 = self.get_first_latent_variables(X, trajectory_indices)
        return self.first_point_distribution.log_prob(X_1).sum()
