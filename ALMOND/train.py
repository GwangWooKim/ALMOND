from util import *
from eval import *

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import os
import json
import matplotlib.pyplot as plt

def train_VAE(model, optimizer, train_loader, u, save_checkpoint, verbose = 0):
    fix_seed(model.seed)

    lst = []
    lst_ll = []
    lst_kl = []

    def model_forward(x):
        ll_loss, kl_loss, loss = model.VAE_forward(x)
        return ll_loss, kl_loss, loss

    model.train()
    for epoch in tqdm(range(model.epochs), desc='Training Loop'):
        train_loss = 0
        train_ll_loss = 0
        train_kl_loss = 0
        model.epoch = epoch
        for step, x in enumerate(train_loader):

            x = x.to(model.device)

            optimizer.zero_grad()
            ll_loss, kl_loss, loss = model_forward(x)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_ll_loss += ll_loss.item()
            train_kl_loss += kl_loss.item()

        lst.append(train_loss / (step + 1))
        lst_ll.append(train_ll_loss / (step + 1))
        lst_kl.append(train_kl_loss / (step + 1))
    
    model.eval()
    
    plt.plot(lst, label = 'loss')
    plt.plot(lst_ll, label = 'll_loss')
    plt.plot(lst_kl, label = 'kl_loss')
    plt.legend()

    return model

def permuation(X, Z):
    indices = torch.randperm(X.size(0))
    return X[indices], Z[indices]

def train_ALMOND(model, optimizer_1, optimizer_2, train_loader, u, save_checkpoint, verbose = 0):
    
    model.kwargs['ftype'] = 'ALMOND'
    
    fix_seed(model.seed)
    lst = []
    lst_lr = []
    
    # make init Z
    with torch.inference_mode():
        X = train_loader.dataset
        mu, logvar = model.encoder(X.to(model.device))
        mu = mu.expand(-1, model.num_samples, -1)
        logvar = logvar.expand(-1, model.num_samples, -1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        Z = mu + eps * std

    model.train()
    for epoch in tqdm(range(model.warmups), desc='Warmup Loop'):
        X, Z = permuation(X, Z) 
        train_loader_ = DataLoader(TensorDataset(X,Z), batch_size = model.batch_size, shuffle = False)
        
        train_loss = 0
        model.epoch = epoch

        step_size = max(model.step_size / 10.0, model.step_size / pow(epoch + 1.0, 0.5))
        z_lst = []

        for step, (x, z) in enumerate(train_loader_):
            
            x, z = x.to(model.device), z.to(model.device)
            lst_lr.append(optimizer_1.param_groups[0]['lr'])

            optimizer_1.zero_grad()
            z = model.langevin_sampling(x, z, step_size)
            u = model.decoder(z)
            
            loss = torch.mean(model.cal_nll(x, u, model.para))
            loss.backward()
            optimizer_1.step()

            train_loss += loss.item()
            z_lst.append(z.cpu())

        lst.append(train_loss / (step + 1))
        Z = torch.vstack(z_lst)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer_2,
                                        lr_lambda=lambda epoch: 1 / (epoch + 1.0) ** 0.5)

    for epoch in tqdm(range(model.warmups, model.epochs), desc='Main Loop'):
        X, Z = permuation(X, Z) 
        train_loader_ = DataLoader(TensorDataset(X,Z), batch_size = model.batch_size, shuffle = False)
        
        train_loss = 0
        model.epoch = epoch

        step_size = max(model.step_size / 10.0, model.step_size / pow(epoch + 1.0, 0.5))
        z_lst = []

        for step, (x, z) in enumerate(train_loader_):
            
            x, z = x.to(model.device), z.to(model.device)
            lst_lr.append(optimizer_2.param_groups[0]['lr'])

            optimizer_2.zero_grad()
            z = model.langevin_sampling(x, z, step_size)
            u = model.decoder(z)
            
            loss = torch.mean(model.cal_nll(x, u, model.para))
            loss.backward()
            optimizer_2.step()

            train_loss += loss.item()
            z_lst.append(z.cpu())

        scheduler.step()
        lst.append(train_loss / (step + 1))
        Z = torch.vstack(z_lst)

    model.eval()
    
    plt.figure()
    plt.plot(lst, label = 'loss')
    plt.legend()

    plt.figure()
    plt.plot(lst_lr, label = 'lr')
    plt.legend()

    return model

def save_model(model, save_path = None):
    if save_path == None:
        save_path = './test'
    os.makedirs(save_path, exist_ok=True)

    # save state
    torch.save(model.state_dict(), f'{save_path}/model_states.pt')

    # save kwargs
    with open(f'{save_path}/model_kwargs.json', 'w') as f: 
        model.kwargs['device'] = str(model.kwargs['device'])
        model.kwargs['act'] = str(model.kwargs['act'])
        model.kwargs['sigma2'] = str(model.kwargs['sigma2'])
        json.dump(model.kwargs, f, indent=4)

def save_optimizer(optimizer, save_path = None):
    if save_path == None:
        save_path = './test'
    os.makedirs(save_path, exist_ok=True)

    dict_ = optimizer.state_dict()
    dict_['name'] = type(optimizer).__name__

    with open(f'{save_path}/optimizer_kwargs.json', 'w') as f: 
        json.dump(dict_, f, indent=4)

