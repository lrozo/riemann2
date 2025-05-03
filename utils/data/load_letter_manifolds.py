"""
Loads letter manifolds: point clouds in the shape of letters,
lying on R2 or on S2. We project the R2 letters to R3 by rotating
them around the x axis.

The current methods allow for loading the letters "G" and "J" in R2,
R3 and S2.
"""

from pathlib import Path

import pickle
from typing import Tuple, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from utils.data.normalization import standard_normalization

torch.set_default_dtype(torch.float64)


def load_letter_in_r2(
    letter: str = "G",
    exp_id: int = 0,
    noise_level: float = 0.1,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Loads the provided letter (either "G" or "J") and the given
    experiment run (between 0 and 6) as a torch.Tensor lying in
    the Euclidean space R2.

    Parameters
    ----------
    - letter (type: str): either "G" or "J".
    - exp_id (type: int): between 0 and 6, representing the
      experiment run.
    - noise_level (type: float): the deviation of the normal noise
      added to the data. Default=0.1
    - scale_factor (type: float): a scale factor by which we multiply
      all components of the data.

    Returns
    -------
    - data (type: torch.Tensor[dtype=float64] of shape (b, 2)): the
        datapoints over R2.
    """
    base_path = Path(__file__).parent.parent.parent.resolve()
    dataset_path = base_path / "data" 

    with open(dataset_path / f"letter_{letter}_R2_{exp_id}.p", "rb") as fp:
        data = pickle.load(fp, encoding="latin1").T

    noise = noise_level * np.random.normal(0.0, 1.0, data.shape)
    data += noise

    scaling_matrix = scale_factor * np.eye(data.shape[1])
    data = data @ scaling_matrix

    return torch.from_numpy(data).type(torch.float64)


def load_letter_in_r3(
    letter: str = "G",
    exp_id: int = 0,
    rotation_angle: float = np.pi / 4,
    noise_level: float = 0.1,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Loads the provided letter (either "G" or "J") and the given
    experiment run (between 0 and 6) as a torch.Tensor lying in
    the Euclidean space R3.

    These are built by rotating the 2-dimensional letters pi/4
    degrees around the x axis. This rotation lifts the data
    from the xy plane onto R3.

    Parameters
    ----------
    - letter (type: str): either "G" or "J".
    - exp_id (type: int): between 0 and 6, representing the
        experiment run.
    - rotation_angle (type: float): an angle (in radians) w.r.t
        which the data is rotated around the x axis.
    - noise_level (type: float): the standard deviation of the
      normal distribution that governs the noise of the data.
    - scale_factor (type: float): a factor by which to
      scale the data.

    Returns
    -------
    - data (type: torch.Tensor[dtype=float64] of shape (b, 3)): the
        datapoints over R3.
    """
    data = load_letter_in_r2(letter=letter, exp_id=exp_id, noise_level=noise_level)

    # Add another column just with 0s
    data = np.concatenate((data, np.zeros((len(data), 1))), axis=1)

    # Rotate
    rotation_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rotation_angle), -np.sin(rotation_angle)],
            [0.0, np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    data = data @ rotation_matrix

    scale_matrix = scale_factor * np.eye(data.shape[1])
    data = data @ scale_matrix

    return torch.from_numpy(data).type(torch.float64)


def load_letter_in_s2(
    letter: str = "G", exp_id: int = 0, noise_level: float = 0.1
) -> torch.Tensor:
    """
    Loads the provided letter (either "G" or "J") and the given
    experiment run (between 0 and 6) as a torch.Tensor lying on
    the sphere S2.

    Parameters
    ----------
    - letter (type: str): either "G" or "J".
    - exp_id (type: int): between 0 and 6, representing the
        experiment run.

    Returns
    -------
    - data (type: torch.Tensor[dtype=float64] of shape (b, 3)): the
        datapoints over S2.
    """
    base_path = Path(__file__).parent.parent.parent.resolve()
    dataset_path = base_path / "data" 

    with open(dataset_path / f"letter_{letter}_S2_{exp_id}.p", "rb") as fp:
        data = pickle.load(fp, encoding="latin1")

    noise = np.random.normal(0.0, noise_level, size=data.shape)
    data += noise
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    return torch.from_numpy(data).type(torch.float64)


def load_toy_example_data(
    letter_in_r2: str = "J",
    letter_in_s2: str = "G",
    noise_level: float = 0.0,
    split_into_train_test: bool = False,
    test_percentage: float = None,
    random_state_for_split: int = None,
    scale_factor_r2=1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Loads the data used in the toy example experiment. In this
    experiment, we learn a latent space of the letter manifold
    data in the product R2 x S2.


    We unify
    the loading here for two reasons:
    1. Easy replicability: this function returns the exact data
    used in the experiment
    2. Following changes in the dataset used using git.

    Returns
    -------

    - training_data (torch.Tensor of shape (b, 5)): the concatenation
    of the letter manifold data on R2 and S3
    """

    letter_data_in_r2 = torch.cat([load_letter_in_r2(letter=letter_in_r2, noise_level=noise_level, exp_id=exp_id,
                                                     scale_factor=scale_factor_r2) for exp_id in range(7)])

    letter_data_in_s2 = torch.cat([load_letter_in_s2(letter=letter_in_s2, noise_level=noise_level, exp_id=exp_id)
                                   for exp_id in range(7)])

    data = torch.hstack([letter_data_in_r2, letter_data_in_s2])

    if split_into_train_test:
        # Once we settle for the models, we should uncomment
        # this: we will need a deterministic split between the
        # data for the GPLVM-based moedls.
        # assert (
        #     random_state_for_split is not None
        # ), "A random state must be provided for replicability at test time."
        if test_percentage is None:
            test_percentage = 0.2

        data_train, data_test = train_test_split(
            data, test_size=test_percentage, random_state=random_state_for_split
        )

        # This adds a little bit of redundancy, but lets us store
        # the first and last positions for easy interpolation. We
        # query the first and last latent code for visualizing a
        # geodesic.
        data_train = torch.vstack((data[0, :], data_train, data[-1, :]))

        return data_train, data_test
    else:
        return data


def load_toy_example_data_v2(letter_in_r2: str = "J", letter_in_s2: str = "C", noise_level: float = 0.0,
                             split_into_train_test: bool = False, scale_factor_r2=1.0) \
        -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Loads the data used in the toy example experiment. In this
    experiment, we learn a latent space of the letter manifold
    data in the product R2 x S2.


    We unify
    the loading here for two reasons:
    1. Easy replicability: this function returns the exact data
    used in the experiment
    2. Following changes in the dataset used using git.

    Returns
    -------

    - training_data (torch.Tensor of shape (b, 5)): the concatenation
    of the letter manifold data on R2 and S3
    """
    test_trajectory_id = 3  # ID of test trajectory
    # Load Euclidean and 2-sphere data
    letter_data_in_r2 = torch.cat([load_letter_in_r2(letter=letter_in_r2, noise_level=noise_level, exp_id=exp_id,
                                                     scale_factor=scale_factor_r2) for exp_id in range(7)])

    letter_data_in_r2, letter_data_in_r2_mean, letter_data_in_r2_std = standard_normalization(letter_data_in_r2)

    letter_data_in_s2 = torch.cat([load_letter_in_s2(letter=letter_in_s2, noise_level=noise_level, exp_id=exp_id)
                                   for exp_id in range(7)])
    data_train = torch.hstack([letter_data_in_r2, letter_data_in_s2])
    # Compute trajectory indexes to support further data processing
    trajectory_indices = torch.Tensor([load_letter_in_r2(letter=letter_in_r2, noise_level=noise_level, exp_id=exp_id,
                                                         scale_factor=scale_factor_r2).shape[0] for exp_id in range(7)])
    trajectory_indices = torch.cumsum(trajectory_indices, dim=0)

    if split_into_train_test:  # Let's remove the test trajectory
        trajectory_index_start = int(trajectory_indices[test_trajectory_id-2])
        trajectory_index_stop = int(trajectory_indices[test_trajectory_id-1])
        data_test = data_train[trajectory_index_start:trajectory_index_stop, :]
        data_train = torch.cat([data_train[0:trajectory_index_start, :], data_train[trajectory_index_stop:, :]])
        trajectory_indices = torch.cat([trajectory_indices[:test_trajectory_id-1],
                                        trajectory_indices[test_trajectory_id:] -
                                        (trajectory_index_stop - trajectory_index_start)])
        return data_train, data_test, trajectory_indices
    else:
        return data_train, trajectory_indices


if __name__ == "__main__":
    load_toy_example_data()
