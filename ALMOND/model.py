from nll import *

import torch
from torch import nn
import numpy as np

class Custom_Layer(nn.Module):
    def __init__(self, in_features, out_features,
                 act, use_normalization = False, use_skip_connection = False) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_features, out_features)
        self.act = act

        if use_normalization:
            self.norm_layer = nn.LayerNorm([out_features])
        else:
            self.norm_layer = nn.Identity()
        
        if use_skip_connection:
            self.linear_2 = nn.Linear(out_features, out_features, bias = False)
            self.skip_connection_layer = nn.Linear(in_features, out_features)
            self.skip_connection = self.skip_connection_function
        else: 
            self.skip_connection = self.identity_for_pairs
            
    def skip_connection_function(self, x, res):
        return self.linear_2(res) + self.skip_connection_layer(x)

    def identity_for_pairs(self, x, res):
        return res

    def forward(self, x):
        res = self.linear_1(x)
        res = self.norm_layer(res)
        res = self.act(res)
        res = self.skip_connection(x, res)
        return res

class LatentModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)   

        if self.dtype == 'MNIST':
            self.encoder = Encoder(**kwargs)
            self.decoder = Decoder(**kwargs)
            self.cal_nll = nll_bernoulli
            self.para = None

        elif self.dtype == 'copula':
            self.encoder = Encoder(**kwargs)
            self.decoder = Decoder(**kwargs)
            self.cal_nll = nll_poisson_reg
            # fixed effect
            self.beta = nn.Linear(self.fix_dim, 1, bias=self.fix_bias)
            # self.para will be passed into nll_poisson_reg
            self.para = [self.fix_dim, self.rand_dim, self.beta]
            # check dimensionality
            assert self.fix_dim + self.rand_dim + 1 == self.x_dim, "Must be fix_dim + rand_dim + 1 == x_dim!"  
        
        else:
            self.encoder = Encoder(**kwargs)
            self.decoder = Decoder(**kwargs)
            self.cal_nll = nll_normal
            # Note that if U ~ N(0, sigma2) and X = U + N(0, 1), then X = N(0, 1+sigma2) and std(X) = sqrt(1+sigma2)
            # Normalizing X, we have X / std(X) = N(0, 1) = U / std(X) + N(0, 1/var(X))
            # So, the sigma2 for likelihood is 1/var(X), not 1. After training, we obtain a sample of U by decoder(z) * std(X)
            self.para = torch.Tensor(0.5 / self.sigma2).to(self.device)

    def VAE_forward(self, x):
        ## forward
        # x = [batch_size, 1, x_dim]
        # z = [batch_size, num_samples, z_dim]
        # u = [batch_size, num_samples, u_dim]
        mu, logvar = self.encoder(x)
        mu = mu.expand(-1, self.num_samples, -1)
        logvar = logvar.expand(-1, self.num_samples, -1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        u = self.decoder(z)

        # loss
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=2))
        ll_loss = torch.mean(self.cal_nll(x, u, self.para))

        return ll_loss, kl_loss, ll_loss + kl_loss

    def langevin_sampling(self, x, z, step_size):
        # z = [batch_size, num_samples, z_dim]
        # u = [batch_size, num_samples, u_dim]
        sd = torch.sqrt(torch.Tensor([2.0 * step_size]).to(self.device))
        z_ = z.clone()
        z_.requires_grad_(True)
        
        for _ in range(self.burn_in):
            u = self.decoder(z_)
            log_pxz = -torch.sum(self.cal_nll(x, u, self.para))
            log_pz = -0.5 * torch.sum(torch.square(z_))
            target = log_pxz + log_pz # NOT mean!

            grad = torch.autograd.grad(target, z_)[0]
            noise = sd * torch.randn_like(z_).to(self.device)
            z_ = z_ + step_size * grad + noise

        return z_.detach()

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.h_dims = [self.x_dim] + list(reversed(self.h_dims))

        layers = []
        for i in range(len(self.h_dims) - 1):
            layers.append(Custom_Layer(self.h_dims[i], self.h_dims[i+1], self.act))
        
        self.mu = nn.Linear(self.h_dims[-1], self.z_dim)
        self.logvar = nn.Linear(self.h_dims[-1], self.z_dim)

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x : [batch_size, 1, x_dim]
        # mu, logvar : [batch_size, 1, z_dim]
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -20, 20)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)   
        
        self.h_dims = [self.z_dim] + self.h_dims

        layers = []
        for i in range(len(self.h_dims) - 1):
            layers.append(Custom_Layer(self.h_dims[i], self.h_dims[i+1], self.act))
        
        layers.append(nn.Linear(self.h_dims[-1], self.u_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        # z : [batch_size, num_samples, z_dim]
        # u : [batch_size, num_samples, u_dim]
        u = self.decoder(z)
        return u