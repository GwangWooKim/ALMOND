import torch
import torch.nn as nn
import numpy as np
import random
import torch.backends.cudnn as cudnn
import scipy.stats as ss

def fix_seed(seed_value = 42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed_value)

def Activation(activation): 
    if activation == 'PReLU':
        act = nn.PReLU()
    elif activation == 'Softplus':
        act = nn.Softplus()    
    else: 
        Exception("Activation must be one of PReLU or Softplus!")
    return act 

class MixtureOfGaussians(ss.rv_continuous):
    def _pdf(self, x):
        res = 0.4 * ss.norm(loc=0, scale=0.5).pdf(x) + 0.6 * ss.norm(loc=3, scale=0.5).pdf(x)
        return res
    
    def _cdf(self, x):
        res = 0.4 * ss.norm(loc=0, scale=0.5).cdf(x) + 0.6 * ss.norm(loc=3, scale=0.5).cdf(x)
        return res

def cal_distances(generated, u, dtype):
    if dtype == 'normal':
        true = ss.norm(loc=0, scale=0.5)

    if dtype == 'exp':
        true = ss.expon(scale=2)
    
    if dtype == 'mix':
        true = MixtureOfGaussians(a=-np.inf, b=np.inf)

    if dtype == 'copula':
        true = ss.gamma(a = 2, loc=-2, scale=1)
        
    ks = ss.kstest(generated, true.cdf)[0]
    wd = ss.wasserstein_distance(u_values = generated, v_values = u, 
                                 #u_weights = np.ones_like(generated), v_weights = true.pdf(u)
                                 )
    ks = round(ks, 4)
    wd = round(wd, 4)
    return ks, wd

def scatter_hist(x, y, ax, ax_histx, ax_histy, label, color):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, alpha=0.4, color = color)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    bins = 100
    ax_histx.hist(x, bins=bins, label = label, density=True, alpha = 0.4, color = color)
    ax_histy.hist(y, bins=bins, orientation='horizontal', density=True, alpha = 0.4, color = color)