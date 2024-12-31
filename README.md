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
```
$ python main.py -d exp
$ python main.py -d mix
$ python main.py -d copula
```
The other arguments are automatically set to reproduce the ALMOND. 

### Note
<img src="/Results/ALMOND_mix/histogram.png" width="70%" height="70%">

One example that confirms the reliability of my implementation is that the results on mix data are similar to those presented in the original paper. 
My results failed to capture the density in the central region (see the figure). However, the lower right panel of Figure 1 in the original paper also showed similar results.
