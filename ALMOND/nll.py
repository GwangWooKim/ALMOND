import torch

# x = [batch_size, 1, x_dim]
# u = [batch_size, num_samples, u_dim], x_dim = u_dim
# nll = [batch_size, num_samples]

def nll_normal(x, u, para):
    nll = torch.sum(para * torch.square(x - u), dim=-1)   
    return nll

def nll_bernoulli(x, u, para):
    nll = -(u * (x - 1) + torch.log(torch.sigmoid(u.to(torch.float64))))
    nll = torch.sum(nll, dim=-1)
    return nll

def nll_nb(x, u, para):
    para = para.exp().clamp(1e-10)
    nll = -( x * torch.log(u / para) - x * torch.log(1 + u / para) - para * torch.log(1 + u / para) 
    + torch.lgamma(x + para) - torch.lgamma(para) - torch.lgamma(x + 1) )
    nll = torch.sum(nll, dim=-1)
    return nll

# x = [batch_size, 1, fix_dim + rand_dim + 1]
# u = [batch_size, num_samples, rand_dim]
# nll = [batch_size, num_samples]

def nll_bernoulli_reg(data, u, list_):
    fix_dim, rand_dim, beta = list_
    x_end = fix_dim
    w_end = fix_dim + rand_dim
    y_end = fix_dim + rand_dim + 1

    x, w, y = data[:, :, :x_end], data[:, :, x_end:w_end], data[:, :, w_end:y_end]
    y_logit = beta(x) + torch.sum(w * u, dim=2, keepdim=True)
    nll = -(y_logit * (y - 1) + torch.log(torch.sigmoid(y_logit.to(torch.float64))))
    nll = torch.sum(nll, dim=-1)
    return nll

def nll_poisson_reg(data, u, list_):
    fix_dim, rand_dim, beta = list_
    x_end = fix_dim
    w_end = fix_dim + rand_dim
    y_end = fix_dim + rand_dim + 1

    x, w, y = data[:, :, :x_end], data[:, :, x_end:w_end], data[:, :, w_end:y_end]
    linear = beta(x) + torch.sum(w * u, dim=2, keepdim=True)
    linear = torch.clamp(linear, -10, 10)

    nll = torch.exp(linear) - y * linear + torch.lgamma(y + 1)
    return nll.squeeze(-1)