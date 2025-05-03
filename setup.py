from setuptools import setup, find_packages

setup(
    name="Riemann2",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "mayavi",
        "gpytorch",
        "geomstats",
    ],
    python_requires=">=3.9",
)
