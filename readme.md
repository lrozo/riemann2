# Riemann2: Learning Riemannian Submanifolds from Riemannian Data

[![arXiv](assets/arXiv-2412.06264-red.svg)](https://arxiv.org/abs/2503.05540)

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

If you are using Pycharm, you can add these environment variables by adding the following to "Run", "Edit Configurations", in the file to run, click on "Environment", and add the following environment variable:
```
GEOMSTATS_BACKEND=pytorch
```


## Code structure & current status

### Experiments

In `./experiments` you can find two PoC examples, which means it is a good place to start. These include regression and latent variable modeling on $\mathbb{S}^2$ and $\mathbb{R}^2\times \mathbb{S}^2$. 

### RiemannSquared

You can find the main Riemann2 model in `./riemannsequared`, where we provide implementations of Wrapped GPLVMS, which in turn, builds on Wrapped GPs and Wrapped Gaussian distributions. Moreover, we provide implementations of back-constraints models (e.g., in `./riemannsequared/models/wrapped_gplvm.py`), and dynamic prior models like the ones used in Gaussian Process Dynamical Models (GPDM) (see `./riemannsequared/models/gpdm_prior.py`)

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