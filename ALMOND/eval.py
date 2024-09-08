from util import *

import os
import json
import matplotlib.pyplot as plt
# import umap
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize

labelencoder = LabelEncoder()

def eval_(model, u, train_loader, save, title = None, save_path = None):

    if save_path == None:
        save_path = './test'
    
    if save:
        os.makedirs(save_path, exist_ok=True)

    fix_seed(model.seed)
    with torch.inference_mode():
        if model.dtype == 'MNIST':
            eval_MNIST(model, u, train_loader, save, title, save_path)

        elif model.dtype == 'copula':
            eval_copula(model, u, train_loader, save, title, save_path)
        
        elif model.dtype == 'circle':
            eval_circle(model, u, train_loader, save, title, save_path)

        elif model.dim == 1:
            eval_1d(model, u, train_loader, save, title, save_path)
        
        elif model.dim == 2:
            eval_2d(model, u, train_loader, save, title, save_path)

def eval_MNIST(model, u, train_loader, save, title, save_path):
    num_generated = 50
    z = torch.randn(num_generated, model.z_dim).to(model.device)
    generated = torch.sigmoid(model.decoder(z)).detach().cpu().numpy()

    # 50 generated digits
    fig = plt.figure()
    plt.title(title)
    plt.axis('off')
    for i in range(num_generated):
        subplot = fig.add_subplot(5, 10, i + 1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(generated[i].reshape(28, 28), cmap=plt.cm.gray)
    plt.tight_layout()

    if save:
        plt.savefig(f'{save_path}/generated_images.png')
    else:
        plt.show()

def true_lambda(x):
    # phi(t) / (dphi(t)/dt)
    return -0.5 * (x - x**3)

def lambda_estimate(u):
    # phi(t) / (dphi(t)/dt) can be estimated by Kendall’s measure, that is, 
    # phi(t) / (dphi(t)/dt) = t - K_{C}(t) (An Introduction to Copulas, Nelsen, Theorem 4.3.4)
    u1 = u[:, 0]
    u2 = u[:, 1]

    n = len(u1)
    z = np.zeros(n)

    for i in range(n):
        z[i] = np.mean((u1 <= u1[i]) & (u2 <= u2[i]))
    
    res = ss.ecdf(z)
    
    points = np.linspace(0, 1, 400)
    values = points - res.cdf.evaluate(points) # t - K_{C}(t)
    return values

def eval_copula(model, u, train_loader, save, title = None, save_path = None):
    z = torch.randn(train_loader.dataset.size(0), model.z_dim).to(model.device)
    generated = model.decoder(z).detach().cpu().numpy()

    ## dimension-wise histograms
    for i in range(model.u_dim):

        # quantative result
        ks, wd = cal_distances(generated[:, i], u[:, i], model.dtype)
        
        plt.figure()
        max_value = 1.1 * max(plt.hist(u[:, i], bins=100, label='True', alpha = 0.4, density=True, color='C1')[0])
        plt.ylim(0, max_value)
        plt.hist(generated[:, i], bins=100, label='Estimated', alpha = 0.4, density=True, color='C2')
        plt.text(0.99,  0.99, f'KS = {ks} \nWD = {wd}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
        plt.title(title)
        plt.tight_layout()
        if save:
            plt.savefig(f'{save_path}/histgram_{i+1}.png')
        else:
            plt.show()
    
    ## Lambda function estimation
    points = np.linspace(0, 1, 400)
    true = true_lambda(points)
    estimated = lambda_estimate(generated)
    L1 = round((np.abs(true - estimated)/400).sum(), 4)
    L2 = round((np.square(true - estimated)/400).sum(), 4)

    plt.figure()
    plt.plot(points, true, label='True', color='C1')
    plt.plot(points, estimated, label='Estimated', color='C2')
    plt.legend()
    plt.text(0.99,  0.05, f'L1 distance = {L1}', horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.text(0.99,  0.01, f'L2 distance = {L2}', horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(f'{save_path}/lambda_function.png')
    else:
        plt.show()

    # for save distances
    distances = {'ks': [cal_distances(generated[:, i], u[:, i], model.dtype)[0] for i in range(model.z_dim)],
                'wd': [cal_distances(generated[:, i], u[:, i], model.dtype)[1] for i in range(model.z_dim)]}
    distances['L1'] = L1
    distances['L2'] = L2
    if save:
        torch.save(distances, f'{save_path}/dist.pt')

def eval_circle(model, u, train_loader, save, title = None, save_path = None):

    # sample generation
    x = train_loader.dataset.squeeze(1).numpy()
    z = torch.randn(train_loader.dataset.size(0), model.z_dim).to(model.device)
    generated = model.decoder(z).detach().cpu().numpy()

    # for 2d scatter plot
    plt.figure()
    plt.scatter(generated[:, 0], generated[:, 1], label = 'generated')
    plt.scatter(x[:, 0], x[:, 1], label = 'train data')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f'{save_path}/scatterplot.png')
    else:
        plt.show()

    # for 2d hists
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    h1 = ax1.hist2d(x[:, 0], x[:, 1], bins=100, density=True)
    ax1.set_title('train data')
    h2 = ax2.hist2d(generated[:, 0], generated[:, 1], bins=100, density=True)
    ax2.set_title('generated')

    cbar1 = fig.colorbar(h1[3], ax=ax1)
    cbar1.set_label('Density')
    cbar2 = fig.colorbar(h2[3], ax=ax2)
    cbar2.set_label('Density')

    plt.tight_layout()
    if save:
        plt.savefig(f'{save_path}/histogram.png')
    else:
        plt.show()

def eval_1d(model, u, train_loader, save, title = None, save_path = None):

    # sample generation
    z = torch.randn(train_loader.dataset.size(0), model.z_dim).to(model.device)
    generated = model.decoder(z).detach().cpu().numpy()
    generated = generated * train_loader.std + train_loader.mean
    
    # quantative result
    ks, wd = cal_distances(generated.reshape(-1, ), u.reshape(-1, ), model.dtype)
    dist = {'ks' : ks, 'wd' : wd}

    # compact domain check
    if model.dtype == 'exp':
        fp = (generated < 0).sum() / len(generated)
        dist['fp'] = fp

    if save:
        with open(f'{save_path}/distances.json', 'w') as f: 
            json.dump(dist, f, indent=4)

    # histogram
    plt.figure()
    max_value = 1.1 * max(plt.hist(u, bins=100, label='True', alpha = 0.4, density=True, color='C1')[0])
    plt.ylim(0, max_value)
    plt.hist(generated, bins=100, label='Estimated', alpha = 0.4, density=True, color='C2')
    plt.text(0.99,  0.99, f'KS = {ks} \nWD = {wd}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(f'{save_path}/histogram.png')
    else:
        plt.show()

def eval_2d(model, u, train_loader, save, title = None, save_path = None):

    # sample generation
    z = torch.randn(train_loader.dataset.size(0), model.z_dim).to(model.device)
    generated = model.decoder(z).detach().cpu().numpy()
    generated = generated * train_loader.std + train_loader.mean

    # for 2d scatter plot
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    scatter_hist(u[:, 0], u[:, 1], ax, ax_histx, ax_histy, label = 'True', color = 'C1')
    scatter_hist(generated[:, 0], generated[:, 1], ax, ax_histx, ax_histy, label = 'Estimated', color = 'C2')
    if model.dtype == 'exp':
        ax.axvline(x=0, c='r')
        ax.axhline(y=0, c='r')
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.9))
    fig.suptitle(title)
    fig.tight_layout()
    if save:
        plt.savefig(f'{save_path}/scatterplot.png')
    else:
        plt.show()
        
    if model.dtype == 'mix':
        # for 2d hists
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        h1 = ax1.hist2d(u[:, 0], u[:, 1], bins=100, density=True)
        ax1.set_title('true latent')
        h2 = ax2.hist2d(generated[:, 0], generated[:, 1], bins=100, density=True)
        ax2.set_title('generated')

        cbar2 = fig.colorbar(h2[3], ax=ax2)
        cbar2.set_label('Density')

        plt.tight_layout()
        if save:
            plt.savefig(f'{save_path}/histogram.png')
        else:
            plt.show()
        

