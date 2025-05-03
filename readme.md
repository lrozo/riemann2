# Riemann2: Learning Riemannian Submanifolds from Riemannian Data

[![arXiv](https://img.shields.io/badge/arXiv-2503.05540-b31b1b.svg)](https://arxiv.org/abs/2503.05540)

</div>

This repository contains the code implementing `Riemann2` and the PoC experiments in our paper.

## Prerequisites

This code was developed using `python 3.9.6`. You can easily install the dependencies by running:

```
pip install -r requirements.txt
```

The code is linted using `black`, so you might also want to run `pip install black` and set up your IDE to format on save.

### Setting up `geomstats`' backend

Since we use `torch` as a backend, **you need to set up the environment variable**
`GEOMSTATS_BACKEND=pytorch`. Also remember to add the working directory to your `PYTHONPATH`.

If you are using VSCode, you can add these environment variables via your `.vscode/settings.json`:

```json
"env": {
    "GEOMSTATS_BACKEND": "pytorch",
    "PYTHONPATH": "${workspaceFolder}${pathSeparator}${env:PYTHONPATH}"
}
```

Alternatively, you can add the following line in your `.env`:
```
GEOMSTATS_BACKEND=pytorch
```

If you are using Pycharm, you can add these environment variables by adding the following to "Run", "Edit Configurations", in the file to run, click on "Environment", and add the following environment variable:
```
GEOMSTATS_BACKEND=pytorch
```


## Code structure

### Experiments

In `./experiments` you can find our PoC example, which means it is a good place to start exploring the code. These include regression and latent variable modeling on $\mathbb{R}^2\times \mathbb{S}^2$. 

### Pre-trained models

We provide a pre-trained model in `./trained_models` for the example on $\mathbb{R}^2\times \mathbb{S}^2$, with a corresponding discretized (grid-like) latent space to accelerate geodesics computation. If you want to train a new model, you can just change the name of the experiment in the source file. Keep in mind that when once a new model is trained, it is necessary to discretize its latent space (for computing geodesics), which can take some minutes. 

### RiemannSquared

You can find the main Riemann2 model in `./riemannsquared`, where we provide implementations of Wrapped GPLVMS, which in turn, builds on Wrapped GPs and Wrapped Gaussian distributions. Moreover, we provide implementations of back-constraints models (e.g., in `./riemannsquared/models/wrapped_gplvm.py`), and dynamic prior models like the ones used in Gaussian Process Dynamical Models (GPDM) (see `./riemannsquared/models/gpdm_prior.py`)

### Manifolds

In `./manifolds`, we provide ad-hoc vectorized implementations for the hypersphere manifold and the product of manifolds. This is mainly motivated due to the tangent space parametrization chosen in our model. More details can be found in the paper.

### Metrics

Similarly to `./manifolds`, this folder `./metrics` provides ad-hoc implementations of the metrics for the manifolds of interest in this repo. Such implementations follow the same tangent space parametrization discussed in the paper.


## Contributors

Miguel González-Duque, Noémie Jaquier, Leonel Rozo


## Citation

If you found this repository useful, please cite the following.

```
@inproceedings{Rozo2025:Riemann2,
    title={Riemann{$^2$}: Learning {R}iemannian Submanifolds from {R}iemannian Data}, 
    author={Leonel Rozo and Miguel Gonz{\'a}lez-Duque and No{\'e}mie Jaquier and S{\o}ren Hauberg},
    booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)},
    year={2025},
    url={https://openreview.net/forum?id=Ffb2Gt6p3U}
}
```