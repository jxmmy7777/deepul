import torch
from torch import nn
from torch.nn import functional as F

def KL_divergence(mu, log_var):
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()

def gaussian_NLL(mu, log_var, x):
    # This prevents negative infinity in the log variance
    const_term = torch.log(torch.tensor(2 * torch.pi, device=log_var.device, dtype=log_var.dtype))
    
    # This computes the Gaussian negative log likelihood with a diagonal covariance matrix
    negative_log_likelihood = 0.5 * (torch.exp(log_var) + (x - mu) ** 2 / torch.exp(log_var) + const_term)
    return (negative_log_likelihood).sum(dim=-1).mean()

def loss_fn_ELBO(x_mu, x_log_var, x, mu, log_var, mode = "mse", beta = 1):
    
    if mode =="mse":
        reconstruction_loss =  F.mse_loss(x_mu, x, reduction='none').sum(dim=(1,2,3)).mean()
    else:
        reconstruction_loss = gaussian_NLL(x_mu, x_log_var,x)
    KLD = KL_divergence(mu, log_var)
    
    #check nan
    # print(reconstruction_loss, KLD)
    assert not torch.isnan(reconstruction_loss).any()
    assert not torch.isnan(KLD).any()
    return reconstruction_loss, KLD, reconstruction_loss+KLD*beta