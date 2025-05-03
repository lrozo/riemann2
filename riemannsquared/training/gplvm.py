"""
This module contains a generic training function for
Wrapped Gaussian Processes Latent Variable Models.

"""

from pathlib import Path
import os

import torch
import numpy as np

import gpytorch

from geomstats.geometry.euclidean import Euclidean

from riemannsquared.models.exact_euclidean_gplvm import ExactWGPLVM

from riemannsquared.models.wrapped_gplvm import WrappedGPLVM
from riemannsquared.models.wrapped_marginal_log_likelihood import WrappedMarginalLogLikelihood


def train_exact_gplvm(
    egplvm: ExactWGPLVM,
    max_iterations: int = 200,
    learning_rate: float = 1e-1,
    verbose: bool = False,
    save_as: str = None,
    load_saved: bool = False
) -> ExactWGPLVM:
    assert isinstance(
        egplvm.manifold, Euclidean
    ), f"The provided ExactEuclideanGPLVM is not an Euclidean GPLVM (manifold: {type(egplvm.manifold)})"

    egplvm = train_wrapped_gplvm(
        egplvm,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        verbose=verbose,
        save_as=save_as,
        load_saved=load_saved
    )
    return egplvm


def train_wrapped_gplvm(
    wgplvm: WrappedGPLVM,
    max_iterations: int = 200,
    learning_rate: float = 1e-1,
    verbose: bool = False,
    save_as: str = None,
    load_saved: bool = False,
) -> WrappedGPLVM:
    """
    Trains a Wrapped GPLVM on the dataset in which it was instantiated.
    We train the hyperparameters of the Wrapped GP using an Adam optimizer
    and gradient descent (with full batches).

    TODO (ask Leonel): instead of having them here under utils, we could
    have this as a method under WrappedGaussianProcess. What do you think?

    Parameters
    ----------

    - wgplvm (WrappedGPLVM): the Wrapped GPLVM to train.

    - max_iterations (int, optional): the number of epochs used to
        train the hyperparameters of wgplvm. Default=200

    - learning_rate (float, optional):  the learning rate passed to
        the the torch.optim.Adam optimizer. Default=0.1

    - verbose (bool, optional): a boolean flag that governs whether
        we report the loss at each iteration. Default=False

    - save_as (str, optional): the name with which the model will be
        saved. We save it under "trained_models/{save_as}.pt". By
        default, we don't save. Default=None

    - load_saved (bool, optional): if True, load the model saved under
        the save_as name. Default=False

    Returns
    -------

    - trained_wgplvm (WrappedGPLVM): the same Wrapped GPLVM object
      that was passed, but with optimized hyperparameters. If the user
      provided a save_as keyword, the best hyperparameters are loaded.
    """
    # Some hyperparameters/constant values
    root_dir = Path(__file__).parent.parent.parent
    saved_models_dir = root_dir / "trained_models"

    # Setting up the training
    wgplvm.train()

    # Load saved model
    if load_saved and save_as is not None:
        if os.path.isfile(saved_models_dir / f"{save_as}.pt"):
            wgplvm.load_state_dict(torch.load(saved_models_dir / f"{save_as}.pt"))
        else:
            print('File does not exist, training model...')
            load_saved = False

    # Train the model
    if not load_saved: 

        # Defining the optimizer.
        optimizer = torch.optim.Adam(wgplvm.parameters(), lr=learning_rate)

        # Defining the loss function: negative marginal log likelihood.
        marginal_log_likelihood = WrappedMarginalLogLikelihood(
            wgplvm.tangent_space_likelihood, wgplvm
        )

        # We train the hyperparameters of the kernel and
        # likelihood using gradient descent.
        best_loss = np.inf
        for iteration in range(max_iterations):
            optimizer.zero_grad()

            # Predicting the Euclidean tangents according to the
            # current latent variables.
            current_latent_vars = wgplvm.latent_variable()
            current_tangent_prediction = wgplvm(current_latent_vars)
            current_tangent_targets = wgplvm.get_tangent_targets(
                current_latent_vars, wgplvm.training_targets
            )

            # Predicting the average loss and taking a backward step
            loss = -marginal_log_likelihood(
                current_tangent_prediction, current_tangent_targets
            ).mean()
            loss.backward()
            optimizer.step()

            # Reporting when verbose.
            if verbose:
                print(
                    f"Iteration: {iteration:04d}, " f"Loss: {loss.item():.3E}",
                    f" Noise: {wgplvm.likelihood.noise.item():.3E}",
                )

            # Saving upon improvement.
            if save_as is not None and loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(wgplvm.state_dict(), saved_models_dir / f"{save_as}.pt")

        # If we saved, we load the best model
        if save_as is not None:
            wgplvm.load_state_dict(torch.load(saved_models_dir / f"{save_as}.pt"))

    return wgplvm
