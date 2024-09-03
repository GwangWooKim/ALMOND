# ALMOND
A PyTorch implementation of [Adaptive Latent Modeling and Optimization via Neural Networks and Langevin Diffusion (ALMOND)](https://github.com/yixuan/almond)

## Packages
* pytorch
* scipy
* sklearn
* pandas
* matplotlib
* tqdm

## How to use
I was only interested in the `exp`, `mix`, and `glmm` (or `copula`) cases. So there are three methods:

$ python main.py -d exp
$ python main.py -d mix
$ python main.py -d copula

The other arguments are automatically set to reproduce the ALMOND. 
