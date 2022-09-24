import os 
import logging
import torch
import numpy as np
import pickle
import torch.nn.functional as F
from torch_geometric.nn import DataParallel
from torch.utils.data import ConcatDataset
from torch_geometric.data import DataListLoader, DataLoader
from scipy.special import erfinv


def mse(data, p_energies, p_forces, energy_coeff, device):
    """
    Compute the weighted MSE loss for the energies and forces of each batch.
    """
    y = torch.cat([d.y for d in data]).to(device)
    f = torch.cat([d.f for d in data]).to(device)
    energies_loss = torch.mean(torch.square(y - p_energies))
    forces_loss = torch.mean(torch.square(f - p_forces))
    total_loss = (energy_coeff)*energies_loss + (1-energy_coeff)*forces_loss
    return total_loss, energies_loss, forces_loss


def energy_forces_loss(args, data, p_energies, p_forces, device):
    """
    Compute the weighted MSE loss for the energies and forces of each batch.
    """
    if args.loss_fn == "mse":
        return mse(data, p_energies, p_forces, args.energy_coeff, device)

    else:
        raise(NotImplementedError(f'Loss funciton tag "{args.loss_fn}" not implemented'))
        

def train_energy_only(args, model, loader, optimizer, device, clip_value=150):
    """
    Loop over batches and train model
    return: batch-averaged loss over the entire training epoch
    """
    model.train()
    total_e_loss = []

    for data in loader:
        e = model(data)
        y = torch.cat([d.y for d in data]).to(e.device)
        e_loss = F.mse_loss(e.view(-1), y.view(-1), reduction="sum")

        with torch.no_grad():
            total_e_loss.append(e_loss.item())

        e_loss.backward()
        optimizer.step()

    ave_e_loss = sum(total_e_loss)/len(total_e_loss)
    return ave_e_loss


def train_energy_forces(args, model, loader, optimizer, device, clip_value=150):
    """
    Loop over batches and train model
    return: batch-averaged loss over the entire training epoch
    """
    model.train()
    total_ef_loss = []
    total_e_loss, total_f_loss = [], []


    for data in loader:

        optimizer.zero_grad()
        e = model(data)

        concat_loader = DataLoader(data, batch_size=len(data), shuffle=False)
        for d in concat_loader:
            d.to(e.device)
            e_tmp = model.module(d).view(-1)
            f = torch.autograd.grad(e_tmp, d.pos, grad_outputs=torch.ones_like(e_tmp), create_graph=False)[0]

        ef_loss, e_loss, f_loss = energy_forces_loss(args, data, e, f, e.device)

        with torch.no_grad():
            total_ef_loss.append(ef_loss.item())
            total_e_loss.append(e_loss.item())
            total_f_loss.append(f_loss.item())

        ef_loss.backward()
        optimizer.step()

    ave_ef_loss = sum(total_ef_loss)/len(total_ef_loss)
    ave_e_loss = sum(total_e_loss)/len(total_e_loss)
    ave_f_loss = sum(total_f_loss)/len(total_f_loss)
    return ave_ef_loss, ave_e_loss, ave_f_loss



def get_error_distribution(err_list):
    """
    Compute the MAE and standard deviation of the errors in the examine set.
    """
    err_array = np.array(err_list)
    mae = np.average(np.abs(err_array))
    var = np.average(np.square(np.abs(err_array)-mae))
    return mae, np.sqrt(var)


def get_pred_loss(args, model, loader, optimizer, device, val=False):
    """
    Gets the total loss on the test/val datasets.
    If validation set, then return MAE and STD also
    """
    model.eval()
    total_ef_loss = []
    all_errs = []
    
    for data in loader:
        optimizer.zero_grad()

        e = model(data)
        concat_loader = DataLoader(data, batch_size=len(data), shuffle=False)
        for d in concat_loader:
            d.to(e.device)
            e_tmp = model.module(d).view(-1)
            f = torch.autograd.grad(e_tmp, d.pos, grad_outputs=torch.ones_like(e_tmp), create_graph=False)[0]
            cluster_size = d['size']

        ef_loss, e_loss, f_loss = energy_forces_loss(args, data, e, f, e.device)

        with torch.no_grad():
            total_ef_loss.append(ef_loss.item())
        if val == True:
            y = torch.cat([d.y for d in data]).to(device)
            f_true = torch.cat([d.f for d in data]).to(device)
            energies_loss = torch.abs(y - e)
            f_red = torch.mean(torch.abs(f_true - f), dim=1)

            f_mean = torch.zeros_like(e)
            for i in range(len(e)):
                energies_loss[i] /= cluster_sizes[i]
                f_mean[i] = torch.mean(torch.abs(torch.tensor(f_red[torch.sum(cluster_sizes[0:i]):torch.sum(cluster_sizes[0:i+1])])))

            total_err = (args.energy_coeff)*energies_loss + (1-args.energy_coeff)*f_mean
            total_err = total_err.tolist()
            all_errs += total_err
    
    ave_ef_loss = sum(total_ef_loss)/len(total_ef_loss)
    
    if val == False:
        return ave_ef_loss
    
    else:
        mae, stdvae = get_error_distribution(all_errs) 
        return ave_ef_loss, mae, stdvae

def get_pred_eloss(args, model, loader, optimizer, device):
    model.eval()
    total_e_loss = []

    for data in loader:
        #data = data.to(device)
        #data.pos.requires_grad = True
        optimizer.zero_grad()

        e = model(data)
        y = torch.cat([d.y for d in data]).to(e.device)
        e_loss = torch.mean(torch.square(y - e))

        with torch.no_grad():
            total_e_loss.append(e_loss.item())


    ave_e_loss = sum(total_e_loss)/len(total_e_loss)
    return ave_e_loss
