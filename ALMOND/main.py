from dataloader import *
from util import *
from model import *
from train import * 
from eval import * 

import torch
import subprocess
import argparse
import os

import warnings
warnings.filterwarnings('ignore')

# just reproduce the results in the paper ALMOND
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dtype", choices = ['exp' , 'mix', 'copula'], default = 'exp')
parser.add_argument("-s", "--seed", default = 42, type = int)
args = parser.parse_args()

def make_h_dims(dtype):
    if dtype == 'exp':
        return [10]
    elif dtype == 'mix':
        return [50, 100, 50]
    elif dtype == 'copula':
        return [128, 256]
    else:
        raise Exception('dtype error!')
    
def main():

    # for data
    dtype = args.dtype
    dim = 1
    batch_size = 1000
    x, u, train_loader = data_loader(dtype, dim, batch_size)

    ## for model 
    # x_dim = full data dimension
    # h_dims = must be a list 
    # z_dim = latent dimension
    # u_dim = generated dimension
    # ftype = must be one of VAE or ALMOND
    # num_samples = number of samples for Monte Carlo (note that in almond implementation, nz == nchain == nsim)
    # activation = must be one of PReLU or Softplus
    # burn_in = used for MCMC
    # step_size = step size to used for Langevin sampling
    x_dim = x.shape[1]
    h_dims = make_h_dims(dtype)
    z_dim = 5 if dtype == 'copula' else 1
    u_dim = 5 if dtype == 'copula' else 1
    ftype = 'VAE'
    num_samples = 100
    activation = 'Softplus'
    burn_in = 20 if dtype == 'copula' else 10
    step_size = 0.01

    ## for model (necessary only for some models)
    # sigma2 = used for normal likelihood
    # fix_dim = covariate dimension for fixed effects
    # fix_bias = whether to add an intercept term
    # rand_dim = covariate dimension for random effects
    # lambda_1 = a constant for power function
    # lambda_2 = a weight for kl_loss only when lambda_1 is used
    sigma2 = 1 / (train_loader.std)**2 if train_loader.std is not None else np.array([0.01])
    fix_dim = 5
    fix_bias = False 
    rand_dim = u_dim

    ## for train
    # warmups = ALMOND train epochs before SGD
    lr = 1e-4 if dtype == 'copula' else 1e-2
    epochs = 2000 if dtype == 'copula' else 1000
    warmups = 2000 if dtype == 'copula' else 100
    seed = args.seed
    name = subprocess.check_output(['hostname']).decode().replace('\n','')
    device = torch.device('cuda')

    # model
    fix_seed(seed)
    model = LatentModel(
        dtype = dtype, dim = dim, batch_size = batch_size, 
        x_dim = x_dim, h_dims = h_dims, z_dim = z_dim, u_dim = u_dim, 
        ftype = ftype, act = Activation(activation), num_samples = num_samples, 
        burn_in = burn_in, step_size = step_size,
        sigma2 = sigma2, fix_dim = fix_dim, fix_bias = fix_bias, rand_dim = rand_dim,
        lr = lr, epochs = epochs, warmups = warmups, seed = seed, name = name, device = device,
    ).to(device)

    # VAE init
    optimizer = torch.optim.Adam(model.parameters(), lr = model.lr)
    save_optimizer(optimizer, save_path = f'./VAE_{dtype}')

    model = train_VAE(model, optimizer, train_loader, u, save_checkpoint = False, verbose = 0)
    eval_(model, u, train_loader, save = True, save_path = f'./VAE_{dtype}')
    save_model(model, save_path = f'./VAE_{dtype}')

    ## ALMOND
    # if you want to save the information of the optimizations below, use the next code, for example.
    # save_optimizer(optimizer_1, save_path = f'./ALMOND_{dtype}')
    optimizer_1 = torch.optim.Adam(model.parameters(), lr = model.lr)
    optimizer_2 = torch.optim.SGD(model.parameters(), lr = model.lr, momentum=0.1, weight_decay=0.001)
    
    model = train_ALMOND(model, optimizer_1, optimizer_2, train_loader, u, save_checkpoint = False, verbose = 0)
    eval_(model, u, train_loader, save = True, save_path = f'./ALMOND_{dtype}')
    save_model(model, save_path = f'./ALMOND_{dtype}')

if __name__ == '__main__':
    main()