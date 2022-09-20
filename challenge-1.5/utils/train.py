import os 
import logging
import torch
import numpy as np
import pickle
import torch.nn.functional as F
from scipy.special import erfinv

def percent_error(actual, pred, c=1e-6):
    return torch.mean(torch.abs((actual-pred)/(actual+c)))


def percent_error_squared(actual, pred, c=1e-6):
    return torch.mean(torch.square((actual-pred)/(actual+c)))


def mean_squared_error(actual, pred):
    return torch.mean(torch.square(actual-pred))


def mean_absoulute_error(actual, pred):
    return torch.mean(torch.abs(actual-pred))


# don't need to normalize by size because schnet is size extensive
# reference energies not needed because all clusters have O:H in same proportion
def e_loss(actual, pred, c): 
    return percent_error_squared(actual, pred)

    
def f_loss(actual, pred, c):
    return percent_error_squared(actual, pred)


def relative(data, p_energies, p_forces, energy_coeff, c):
    """
    Compute the weighted relative loss for the energies and forces of each batch.
    """
    energies_loss = e_loss(data.y, p_energies, c)
    forces_loss = f_loss(data.f, p_forces, c)
    total_loss = (energy_coeff)*(energies_loss) + (1-energy_coeff)*(forces_loss)
  
    return total_loss, energies_loss, forces_loss


def relative_energy(data, p_energies, p_forces, energy_coeff, c):
    """
    Compute the weighted relative loss for the energies and forces of each batch.
    """
    energies_loss = e_loss(data.y, p_energies, c)
    forces_loss = torch.mean(torch.square(data.f - p_forces))
    total_loss = (energy_coeff)*(energies_loss) + (1-energy_coeff)*(forces_loss)
  
    return total_loss, energies_loss, forces_loss


def relative_gradient(data, p_energies, p_forces, energy_coeff, c):
    """
    Compute the weighted relative loss for the energies and forces of each batch.
    """
    energies_loss = torch.mean(torch.square(data.y - p_energies))
    forces_loss = f_loss(data.f, p_forces, c)
    total_loss = (energy_coeff)*(energies_loss) + (1-energy_coeff)*(forces_loss)
  
    return total_loss, energies_loss, forces_loss


def mse(data, p_energies, p_forces, energy_coeff):
    """
    Compute the weighted MSE loss for the energies and forces of each batch.
    """
    energies_loss = torch.mean(torch.square(data.y - p_energies))
    forces_loss = torch.mean(torch.square(data.f - p_forces))
    total_loss = (energy_coeff)*energies_loss + (1-energy_coeff)*forces_loss
    return total_loss, energies_loss, forces_loss


def energy_forces_loss(args, data, p_energies, p_forces, energy_coeff, c):
    """
    Compute the weighted MSE loss for the energies and forces of each batch.
    """
    if args.loss_fn == "mse":
        return mse(data, p_energies, p_forces, energy_coeff)

    elif args.loss_fn == "relative":
        return relative(data, p_energies, p_forces, energy_coeff, c)

    elif args.loss_fn == "relative_energy":
        return relative_energy(data, p_energies, p_forces, energy_coeff, c)

    elif args.loss_fn == "relative_grad":
        return relative_gradient(data, p_energies, p_forces, energy_coeff, c)
    else:
        loss_fn = args.loss_fn
        raise(NotImplementedError(f'Loss funciton tag "{loss_fn}" not implemented'))
        

def train_energy_only(args, model, loader, optimizer, energy_coeff, device, clip_value=150):
    """
    Loop over batches and train model
    return: batch-averaged loss over the entire training epoch
    """
    model.train()
    total_e_loss = []

    for data in loader:
        data = data.to(device)
        e = model(data)
        e_loss = F.mse_loss(e.view(-1), data.y.view(-1), reduction="sum")

        with torch.no_grad():
            total_e_loss.append(e_loss.item())

        e_loss.backward()
        optimizer.step()

    ave_e_loss = sum(total_e_loss)/len(total_e_loss)
    return ave_e_loss


def train_energy_forces(args, model, loader, optimizer, energy_coeff, device, c, clip_value=150):
    """
    Loop over batches and train model
    return: batch-averaged loss over the entire training epoch 
    """
    model.train()
    total_ef_loss = []
    total_e_loss, total_f_loss = [], []

    for data in loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()
        e = model(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=True)[0]

        ef_loss, e_loss, f_loss = energy_forces_loss(args, data, e, f, energy_coeff, c)
        
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


def get_idx_to_add(net, examine_loader, optimizer,
                   mae, std, energy_coeff, 
                   split_file, al_step, device, min_nonmin,
                   max_to_add=0.15, error_tolerance=0.15,
                   savedir = './'):
    """
    Computes the normalized (by cluster size) errors for all entries in the examine set. It will add a max of
    max_to_add samples that are p < 0.15.
    """
    net.eval()
    all_errs = []
    for data in examine_loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()

        e = net(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]
        energies_loss = torch.abs(data.y - e)
        f_red = torch.mean(torch.abs(data.f - f), dim=1)
        
        f_mean = torch.zeros_like(e)
        cluster_sizes = data['size'] #data.size
        for i in range(len(e)):            #loop over all clusters in batch
            energies_loss[i] /= cluster_sizes[i]
            f_mean[i] = torch.mean(torch.abs(torch.tensor(f_red[torch.sum(cluster_sizes[0:i]):torch.sum(cluster_sizes[0:i+1])]))).clone().detach()
        
        total_err = (energy_coeff)*energies_loss + (1-energy_coeff)*f_mean
        total_err = total_err.tolist()
        all_errs += total_err
    
    with open(os.path.join(savedir, f'error_distribution_alstep{al_step}_{min_nonmin}.pkl'), 'wb') as f:
        pickle.dump(all_errs, f)    

    S = np.load(os.path.join(savedir, split_file))
    examine_idx = S["examine_idx"].tolist()
    
    cutoff = erfinv(1-error_tolerance) * std + mae
    n_samples_to_add = int(len(all_errs)*max_to_add)
    idx_highest_errors = np.argsort(np.array(all_errs))[-n_samples_to_add:]
    idx_to_add = [examine_idx[idx] for idx in idx_highest_errors if all_errs[idx]>=cutoff]
    
    return idx_to_add


def get_pred_loss(args, model, loader, optimizer, energy_coeff, device, c, val=False):
    """
    Gets the total loss on the test/val datasets.
    If validation set, then return MAE and STD also
    """
    model.eval()
    total_ef_loss = []
    all_errs = []
    
    for data in loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()

        e = model(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]

        ef_loss, e_loss, f_loss = energy_forces_loss(args, data, e, f, energy_coeff, c)
        with torch.no_grad():
            total_ef_loss.append(ef_loss.item())
        if val == True:
            energies_loss = torch.abs(data.y - e)
            f_red = torch.mean(torch.abs(data.f - f), dim=1)

            f_mean = torch.zeros_like(e)
            cluster_sizes = data['size'] #data.size
            for i in range(len(e)):            #loop over all clusters in batch
                energies_loss[i] /= cluster_sizes[i]
                f_mean[i] = torch.mean(torch.abs(torch.tensor(f_red[torch.sum(cluster_sizes[0:i]):torch.sum(cluster_sizes[0:i+1])])))

            total_err = (energy_coeff)*energies_loss + (1-energy_coeff)*f_mean
            total_err = total_err.tolist()
            all_errs += total_err
    
    ave_ef_loss = sum(total_ef_loss)/len(total_ef_loss)
    
    if val == False:
        return ave_ef_loss
    
    else:
        mae, stdvae = get_error_distribution(all_errs) #MAE and STD from EXAMINE SET
        return ave_ef_loss, mae, stdvae

def get_pred_eloss(args, model, loader, optimizer, energy_coeff, device):
    model.eval()
    total_e_loss = []

    for data in loader:
        data = data.to(device)
        data.pos.requires_grad = True
        optimizer.zero_grad()

        e = model(data)
        e_loss = torch.mean(torch.square(data.y - e))

        with torch.no_grad():
            total_e_loss.append(e_loss.item())


    ave_e_loss = sum(total_e_loss)/len(total_e_loss)
    return ave_e_loss
