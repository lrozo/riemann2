import numpy as np
from dtaidistance import dtw_ndim
from typing import List


def get_dynamic_time_warping_distance_list(demonstrations, reproduced_trajectories) -> List[float]:
    """
    Compute the dynamic time warping distance between demonstration trajectories and reproduced trajectories.
    The reproduced trajectories start from demonstrations' initial points.
    :param manifold_type: the type of manifold, e.g. 'sphere', 'euclidean'
    ;param demonstration_name: the name of demonstration, e.g. 'NShape'
    """

    # Project the data on the manifold if the model trained in Euclidean space
    # if manifold_type == "projected_euclidean":
    #     reproduced_trajectories = reproduced_trajectories / np.linalg.norm(reproduced_trajectories, axis=2)[:, :, None]

    nb_demos = len(demonstrations)
    nb_repros = len(reproduced_trajectories)

    dwt_list = []
    for i_fold in range(nb_demos):  # i_fold as test dataset, iterate through all trained models

        for ind_repro in range(nb_repros):  # iterate through all reproduced trajectories
            # print('i_fold {}: ind_repro {}'.format(i_fold, ind_repro))
            repro0 = reproduced_trajectories[ind_repro]

            # compute the index for the correponding demo trajectory
            # j0 = [j for j in range(nb_demos) if j != i_fold][ind_repro]
            # j0 = ind_repro
            j0 = i_fold

            # pick the correponding demo trajectory
            demo0 = demonstrations[j0]

            max_size = max(repro0.shape[0], demo0.shape[0])

            # pad data such that two trajectories have the same length
            repro0_pad = np.pad(repro0, ((0, max_size - repro0.shape[0]), (0, 0)), 'edge')
            demo0_pad = np.pad(demo0, ((0, max_size - demo0.shape[0]), (0, 0)), 'edge')

            # compute dynamic time warping distance
            d = dtw_ndim.distance(repro0_pad, demo0_pad)
            dwt_list.append(d)

    return dwt_list
