"""
Here we implement some wrappers that make computing geodesics easy
via Stochman [1]. Stochman is a Python package that has interfaces
for Manifolds and EmbeddedManifolds, which makes it easy to compute
energy-minimizing curves given a class that implements either an
"embed" method, or a "metric" method.

[1] https://github.com/MachineLearningLifeScience/stochman
"""
from pathlib import Path
import pickle
from typing import List, Union

import torch

from stochman.manifold import Manifold, EmbeddedManifold
from stochman.discretized_manifold import DiscretizedManifold
from stochman.curves import BasicCurve, CubicSpline

from riemannsquared.models.exact_euclidean_gplvm import ExactWGPLVM

from riemannsquared.models.wrapped_gplvm import WrappedGPLVM


class DiscreteGPLVMManifoldWrapper(DiscretizedManifold):
    def __init__(
        self,
        gplvm: Union[ExactWGPLVM, WrappedGPLVM],
        grid: List[torch.Tensor],
        scale_for_covariance_term: float = 1.0,
    ):
        super().__init__()

        self.gplvm = gplvm
        self.grid = grid
        self.scale_for_covariance_term = scale_for_covariance_term

    def fit(self, batch_size: int = 4):
        return super().fit(self, self.grid, batch_size=batch_size)

    def metric(self, c: torch.Tensor, return_deriv=False):
        return self.gplvm.predict_pullback_metric(
            c, scale_for_covariance_term=self.scale_for_covariance_term
        )

    def save_discretized_manifold(self, filepath: Path):
        obj_to_save = {
            "G": self.G,
            "grid": self.grid,
            "grid_size": self.grid_size,
            "__metric__": self.__metric__,
        }

        with open(filepath, "wb") as fp:
            pickle.dump(obj_to_save, fp)

    @classmethod
    def from_path(
        cls,
        gplvm: Union[ExactWGPLVM, WrappedGPLVM],
        filepath: Path,
        scale_for_covariance_term: float = 1.0,
    ):
        with open(filepath, "rb") as fp:
            saved_obj = pickle.load(fp)

        # Creating an instance
        grid = saved_obj["grid"]
        gplvm_as_manifold = cls(
            gplvm, grid, scale_for_covariance_term=scale_for_covariance_term
        )

        # Loading the saved graph
        gplvm_as_manifold.G = saved_obj["G"]
        gplvm_as_manifold.grid_size = saved_obj["grid_size"]
        gplvm_as_manifold.__metric__ = saved_obj["__metric__"]

        return gplvm_as_manifold


class GPLVMManifoldWrapper(Manifold):
    def __init__(self, gplvm: Union[ExactWGPLVM, WrappedGPLVM],scale_for_covariance_term: float = 1.0,):
        super().__init__()

        self.gplvm = gplvm
        self.scale_for_covariance_term = scale_for_covariance_term

    def metric(self, c: torch.Tensor, return_deriv=False):
        return self.gplvm.predict_pullback_metric(c, scale_for_covariance_term=self.scale_for_covariance_term)


def build_discretized_manifold_from_gplvm(
    wgplvm: WrappedGPLVM,
    padding_percentage: float = 0.05,
    grid_resolution: int = 75,
    grid_file_name: str = None,
    from_cache: bool = True,
    save_cache: bool = True,
    batch_size: int = 4,
    scale_for_covariance_term: float = 1.0,
) -> DiscreteGPLVMManifoldWrapper:
    """
    TODO: write docs
    """
    ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
    CACHED_GRIDS_PATH = ROOT_DIR / "data" / "cached_grids"

    if grid_file_name is None:
        # Using a default name
        grid_file_name = f"wgplvm_motion_synthesis_grid_{grid_resolution}"

    # The path to load from/save to.
    grid_file_path = CACHED_GRIDS_PATH / (grid_file_name + ".pkl")

    if from_cache and grid_file_path.exists():
        # Loading up the object from the cache.
        wgplvm_as_manifold = DiscreteGPLVMManifoldWrapper.from_path(wgplvm, grid_file_path)
        return wgplvm_as_manifold

    else:
        # Building it from scratch and using a default name.
        # Setting up the limits of the grid.
        min_latent_var = wgplvm.latent_variable().min().item()
        max_latent_var = wgplvm.latent_variable().max().item()
        padding = (max_latent_var - min_latent_var) * padding_percentage
        limits = [min_latent_var - padding, max_latent_var + padding]

        # Building the manifold and fitting it.
        wgplvm_as_manifold = DiscreteGPLVMManifoldWrapper(wgplvm, [torch.linspace(*limits, grid_resolution),
                                                                   torch.linspace(*limits, grid_resolution)],
                                                          scale_for_covariance_term=scale_for_covariance_term)
        wgplvm_as_manifold.fit(batch_size=batch_size)

        # Caching it
        if save_cache:
            wgplvm_as_manifold.save_discretized_manifold(grid_file_path)

        return wgplvm_as_manifold


def geodesic_minimizing_energy(manifold, p0=None, p1=None, init_curve=None, 
                               optimizer=torch.optim.Adam, lr=1e-1, max_iter=150, eval_grid=20):
    """
    Compute a geodesic curve connecting two points by minimizing its energy.
    Copied and adapted from Stochman

    Mandatory inputs:
        curve:      A curve object representing a curve with fixed end-points.
                    When the function returns, this object has been updated to
                    be a geodesic curve.
        manifold:   A manifold object representing the space over which the
                    geodesic is defined. This object must provide a
                    'curve_energy' function through which pytorch can
                    back-propagate.

    Optional inputs:
        optimizer:  Choice of iterative optimizer.
                    Default: torch.optim.Adam
        max_iter:   The maximum number of iterations of the optimizer.
                    Default: 150
        eval_grid:  The number of points along the curve where
                    energy is evaluated.
                    Default: 20

    Output:
        success:    True if the algorithm converged, False otherwise.

    Example usage:
    S = Sphere()
    p0 = torch.tensor([0.1, 0.1]).reshape((1, -1))
    p1 = torch.tensor([0.3, 0.7]).reshape((1, -1))
    C = CubicSpline(begin=p0, end=p1, num_nodes=8, requires_grad=True)
    geodesic_minimizing_energy(C, S)
    """
    if init_curve is None:
        curve = CubicSpline(p0, p1)
    else:
        curve = init_curve
        # curve.begin = p0
        # curve.end = p1

    # Initialize optimizer and set up closure
    alpha = torch.linspace(0, 1, eval_grid, dtype=curve.begin.dtype, device=curve.device)
    opt = optimizer(curve.parameters(), lr=lr)

    thresh = 1e-4

    for k in range(max_iter):
        opt.zero_grad()
        loss = manifold.curve_energy(curve(alpha)).mean()
        loss.backward()
        opt.step()
        max_grad = max([p.grad.abs().max() for p in curve.parameters()])

        print(f"Iteration {k}: {loss}")

        if max_grad < thresh:
            break
        # if k % (max_iter // 10) == 0:
        #    curve.constant_speed(manifold)
    # curve.constant_speed(manifold)
    return curve, max_grad < thresh


def fit_to_curve(points_to_fit, p0=None, p1=None, init_curve=None, 
                 optimizer=torch.optim.Adam, lr=1e-1, max_iter=150, verbose=False):
    """
    Compute a geodesic curve connecting two points by minimizing its energy.
    Copied and adapted from Stochman

    Mandatory inputs:
        curve:      A curve object representing a curve with fixed end-points.
                    When the function returns, this object has been updated to
                    be a geodesic curve.
        manifold:   A manifold object representing the space over which the
                    geodesic is defined. This object must provide a
                    'curve_energy' function through which pytorch can
                    back-propagate.

    Optional inputs:
        optimizer:  Choice of iterative optimizer.
                    Default: torch.optim.Adam
        max_iter:   The maximum number of iterations of the optimizer.
                    Default: 150
        eval_grid:  The number of points along the curve where
                    energy is evaluated.
                    Default: 20

    Output:
        success:    True if the algorithm converged, False otherwise.

    Example usage:
    S = Sphere()
    p0 = torch.tensor([0.1, 0.1]).reshape((1, -1))
    p1 = torch.tensor([0.3, 0.7]).reshape((1, -1))
    C = CubicSpline(begin=p0, end=p1, num_nodes=8, requires_grad=True)
    geodesic_minimizing_energy(C, S)
    """
    if init_curve is None:
        curve = CubicSpline(p0, p1)
    else:
        curve = init_curve
        # curve.begin = p0
        # curve.end = p1

    # Initialize optimizer and set up closure
    eval_grid = points_to_fit.shape[0]
    alpha = torch.linspace(0, 1, eval_grid, dtype=curve.begin.dtype, device=curve.device)
    opt = optimizer(curve.parameters(), lr=lr)

    thresh = 1e-4

    for k in range(max_iter):
        opt.zero_grad()
        loss = torch.mean((points_to_fit - curve(alpha))**2)  
        loss.backward()
        opt.step()
        max_grad = max([p.grad.abs().max() for p in curve.parameters()])

        if verbose:
            print(f"Iteration {k}: {loss}")

        if max_grad < thresh:
            break
        # if k % (max_iter // 10) == 0:
        #    curve.constant_speed(manifold)
    # curve.constant_speed(manifold)
    return curve, max_grad < thresh


class DensityMetricWrapper(Manifold):
    def __init__(self, data, sigma):
        super().__init__()
        self.data = data
        self.sigma = sigma

    def density(self, x):
        """
        Evaluate a kernel density estimate at x.

        Parameters:
        x: points where the density is evaluated. Dimensions (num_points)x(data_dim)
        """
        N, D = x.shape
        M, _ = self.data.shape
        sigma2 = self.sigma**2
        normalization = (2 * 3.14159)**(D/2) * self.sigma**D  # scalar
        distances = torch.cdist(x, self.data)  # NxM
        K = torch.exp(-0.5 * distances**2 / sigma2) / normalization  # NxM
        p = torch.sum(K, dim=1)  # N
        return p
    
    def metric(self, c: torch.Tensor, return_deriv=False):
        p = self.density(c)
        D = c.shape[-1]
        return 1.0/p[:, None].repeat(1, D)

