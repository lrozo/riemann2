import os
import numpy as np
import scipy.io as sp
import torch
from mayavi import mlab

from geomstats.geometry.hypersphere import Hypersphere

from utils.visualization.manifolds.sphere_utils import plot_sphere_mayavi


if __name__ == "__main__":
    sphere_manifold = Hypersphere(2)
    sphere_origin = torch.from_numpy(np.array([1., 0., 0.]))

    # Load data and project them on the sphere
    current_path = os.path.dirname(__file__)
    letter_data = sp.loadmat(current_path + '/2D_letters/C.mat')['demos']
    num_demos = len(letter_data[0])
    num_demos = 7
    demos_data = [letter_data[0, i][0, 0][0] for i in range(num_demos)]
    data_euclidean = np.hstack(demos_data).T

    nb_data = data_euclidean.shape[0]
    data_euclidean = np.hstack((np.zeros((nb_data, 1)), data_euclidean))
    data_euclidean = torch.from_numpy(data_euclidean)
    data = sphere_manifold.metric.exp(0.1 * data_euclidean, sphere_origin)

    data_np = data.detach().numpy()
    for i in range(num_demos): 
        data_exp = data_np[i*200:(i+1)*200]
        data_exp.dump(current_path + '/letter_C_S2_' + str(i) + '.p')

    # # Plot data
    # mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(700, 700))
    # fig = mlab.gcf()
    # mlab.clf()
    # plot_sphere_mayavi(radius=1.0, opacity=0.2, figure=fig)
    # for n in range(nb_data):
    #     mlab.points3d(data[n, 0], data[n, 1], data[n, 2], color=(0., 0., 0.), scale_factor=0.03)
    # mlab.show()