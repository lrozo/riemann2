import torch
from torch.optim.adam import Adam
from gpytorch.mlls.added_loss_term import AddedLossTerm


def stress(distances: torch.Tensor, graph_distances: torch.Tensor) -> torch.Tensor:
    """
    Computes the stress
    """
    distances_diff = graph_distances - distances
    i, j = torch.triu_indices(*distances_diff.shape, offset=1)

    return torch.pow(torch.triu(distances_diff, diagonal=1), 2)[i, j]


def euclidean_squared_distance(tensor1, tensor2):
    # Ensure both tensors have the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must have the same shape")

    squared_diff = (tensor1 - tensor2) ** 2  # Calculate element-wise squared differences
    squared_distances = squared_diff.sum(dim=1)  # Sum along the appropriate dimensions to get squared distances

    return squared_distances


def pairwise_squared_euclidean_distances(data_tensor):
    num_tensors, dim = data_tensor.size()

    # Expand the tensor to be of shape (num_tensors, num_tensors, dim)
    expanded_tensor = data_tensor.unsqueeze(0).expand(num_tensors, -1, -1)

    squared_diff = (expanded_tensor - expanded_tensor.transpose(0, 1)) ** 2  # Compute the squared differences
    squared_distances = squared_diff.sum(dim=2)  # Sum along the dimension to get squared distances

    return squared_distances


def pairwise_squared_distances(data_tensor, squared_distance_function):
    num_tensors, dim = data_tensor.size()

    # Expand the tensor to be of shape (num_tensors, num_tensors, dim)
    expanded_tensor = data_tensor.unsqueeze(0).expand(num_tensors, -1, -1)

    squared_distances = squared_distance_function(expanded_tensor, expanded_tensor.transpose(0, 1))

    return squared_distances


class StressLossTermExactMLL(AddedLossTerm):
    """
    Class to compute cost regularizer (coming from a prior on latent variables) based on the stress loss (Eq. 10 in [1])

    [1] C. Cruceru. "Computationally Tractable Riemannian Manifolds for Graph Embeddings." AAAI, 2021

    Parameters
    ----------
    graph_distances: Graph distance matrix for all pairs of shape poses
    loss_scale: Constant to control magnitude of distortion loss
    """
    def __init__(self, ambient_distances, latent_distance_function, loss_scale=1.0):
        super().__init__()
        self.ambient_distances = ambient_distances  # As this does not change for the optimization, it is a parameter
        self.latent_distance = latent_distance_function
        self.loss_scale = loss_scale

    def loss(self, x, **kwargs):
        """
        Implements stress loss according to Eq. 10 in [1].

        Parameters
        ----------
        x: Optimization variable (i.e. latent variables in GPLVM)
        kwargs: Additional arguments for loss function

        Returns
        -------
        stress_loss: Sum over the squared difference between the graph and manifold distances as defined in [1]
                     The negative stress is return to comply with GPytorch that maximizes added terms.
        """
        distances = self.latent_distance(x)
        stress_loss = self.loss_scale * torch.mean(stress(distances, self.ambient_distances))

        return - stress_loss


class EuclideanStressLossTermExactMLL(StressLossTermExactMLL):
    """
    Class to compute cost regularizer (coming from a prior on latent variables) based on the stress loss (Eq. 10 in [1])

    [1] C. Cruceru. "Computationally Tractable Riemannian Manifolds for Graph Embeddings." AAAI, 2021

    Parameters
    ----------
    ambient_distances:
    loss_scale: Constant to control magnitude of distortion loss
    """
    def __init__(self, ambient_distances, loss_scale=1.0):

        super().__init__(ambient_distances, pairwise_squared_euclidean_distances, loss_scale)


def trajectories_stress_loss_initialization(training_targets, latent_dim, trajectory_indexes, squared_distance_function):
    """
    Stress initialization aims at having latent variables whose distance closely matches the distances of observations in the ambient space.

    Parameters
    ----------
    training_targets: observations  [nb_data x dimension]
    latent_dim: dimension of the latent space
    trajectory_indexes: list of index of trajectory starting points
    """
    
    n_points, *_ = training_targets.shape

    # 0. Extract start and end points from training data to only apply initialization on extreme points
    #   of the trajectory. Later, we will interpolate linearly between them
    ambient_start_trajectory_points = torch.stack(([training_targets[int(trajectory_indexes[i]), :]
                                            for i in range(len(trajectory_indexes) - 1)]))
    ambient_start_trajectory_points = torch.cat([training_targets[0, :][None],
                                                    ambient_start_trajectory_points])
    ambient_end_trajectory_points = torch.stack(([training_targets[int(trajectory_indexes[i]) - 1, :]
                                            for i in range(len(trajectory_indexes))]))
    ambient_start_end_trajectory_points = torch.cat([ambient_start_trajectory_points,
                                                        ambient_end_trajectory_points])
    ambient_distances = pairwise_squared_distances(ambient_start_end_trajectory_points, squared_distance_function)
    ambient_distances2 = pairwise_squared_euclidean_distances(ambient_start_end_trajectory_points)


    # 1. We still do a first pre-initialization using PCA
    _, _, V = torch.pca_lowrank(training_targets.flatten(start_dim=1), q=latent_dim)
    initial_latent_variables = torch.nn.Parameter(torch.matmul(training_targets.flatten(start_dim=1),
                                                                V[:, :latent_dim]))
    latent_start_trajectory_points = torch.stack(([initial_latent_variables[int(trajectory_indexes[i]), :]
                                            for i in range(len(trajectory_indexes) - 1)]))
    latent_start_trajectory_points = torch.cat([initial_latent_variables[0, :][None],
                                                latent_start_trajectory_points])
    latent_end_trajectory_points = torch.stack(([initial_latent_variables[int(trajectory_indexes[i]) - 1, :]
                                            for i in range(len(trajectory_indexes))]))
    latent_start_end_trajectory_points = torch.cat([latent_start_trajectory_points,
                                                    latent_end_trajectory_points])
    latent_start_end_trajectory_points = torch.nn.Parameter(latent_start_end_trajectory_points)

    # 2. Now we carry out the stress-based initialization
    stress = EuclideanStressLossTermExactMLL(ambient_distances)

    optim = Adam([{"params": latent_start_end_trajectory_points}], lr=0.01)
    optim_steps = 1000

    for step in range(optim_steps):
        optim.zero_grad()
        loss = -stress.loss(latent_start_end_trajectory_points)
        loss.backward()
        optim.step()
        if not step % 100:
            print(f"Iter {step + 1}/{optim_steps}: {loss.item()}")
    print(f"Iter {step + 1}/{optim_steps}: {loss.item()}")

    # Linear interpolation between start and end trajectory points in the latent space
    initial_latent_variables = torch.zeros(n_points, latent_dim, dtype=training_targets.dtype)

    # weights = torch.linspace(0., 1., int(n_points / len(trajectory_indexes)))
    weights = torch.linspace(0., 1., int(trajectory_indexes[0]))
    initial_latent_variables[0:int(trajectory_indexes[0])] = \
        torch.lerp(latent_start_end_trajectory_points[0][None],
                    latent_start_end_trajectory_points[len(trajectory_indexes)][None],
                    weights[:, None])

    for i in range(len(trajectory_indexes)-1):
        start = int(trajectory_indexes[i])
        end = int(trajectory_indexes[i+1])
        weights = torch.linspace(0., 1., end-start)
        initial_latent_variables[start:end] = torch.lerp(latent_start_end_trajectory_points[i+1][None],
                    latent_start_end_trajectory_points[i + 1 + len(trajectory_indexes)][None],
                    weights[:, None])

    initial_latent_variables = torch.nn.Parameter(initial_latent_variables)

    return initial_latent_variables
