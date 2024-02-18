from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch
import numpy as np

def train_and_evaluate(train_loader, test_loader, model, hyperparams, optimizer, loss_fn, scheduler=None, save_checkpoint=True, checkpoint_path=None, device=None, debug_mode=False):
    num_epochs = 1 if debug_mode else hyperparams['num_epochs']

    train_losses = []
    test_losses = []

    # Modified to store additional details
    train_reconstruction_losses = []
    train_KL_divergences = []
    test_reconstruction_losses = []
    test_KL_divergences = []

    def evaluate_test_data():
        model.eval()
        total_test_loss = 0
        total_reconstruction_loss = 0
        total_KL_divergence = 0
        with torch.no_grad():
            for i, batch_inputs in enumerate(test_loader):
                batch_inputs = batch_inputs[0].to(device)
                x_mu, x_log_var, mu, log_var = model(batch_inputs)
                reconstruction_loss, KLD, test_loss = loss_fn(x_mu, x_log_var, batch_inputs, mu, log_var)
                total_test_loss += test_loss.item()
                total_reconstruction_loss += reconstruction_loss.item()
                total_KL_divergence += KLD.item()
        avg_test_loss = total_test_loss / len(test_loader)
        avg_reconstruction_loss = total_reconstruction_loss / len(test_loader)
        avg_KL_divergence = total_KL_divergence / len(test_loader)
        return avg_reconstruction_loss, avg_KL_divergence, avg_test_loss

    initial_reconstruction_loss, initial_KL_divergence, initial_test_loss = evaluate_test_data()
    test_losses.append(initial_test_loss)
    test_reconstruction_losses.append(initial_reconstruction_loss)
    test_KL_divergences.append(initial_KL_divergence)

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        for i, batch_inputs in enumerate(train_loader):
            batch_inputs = batch_inputs[0].to(device)
            optimizer.zero_grad()

            x_mu,x_log_var, mu, log_var = model(batch_inputs)
            reconstruction_loss, KLD, loss = loss_fn(x_mu, x_log_var, batch_inputs, mu, log_var)
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # Store losses separately
            train_losses.append(loss.item())
            train_reconstruction_losses.append(reconstruction_loss.item())
            train_KL_divergences.append(KLD.item())

            if debug_mode:
                break

        epoch_reconstruction_loss, epoch_KL_divergence, epoch_test_loss = evaluate_test_data()
        test_losses.append(epoch_test_loss)
        test_reconstruction_losses.append(epoch_reconstruction_loss)
        test_KL_divergences.append(epoch_KL_divergence)

    if save_checkpoint and checkpoint_path:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_test_loss,
        }, f"{checkpoint_path}_epoch_{epoch}.pth")

    # Adjust return statement to include all metrics
    train_losses_stacked = np.stack((np.array(train_losses),
                                    np.array(train_reconstruction_losses),
                                    np.array(train_KL_divergences)), axis=1)

    test_losses_stacked = np.stack((np.array(test_losses),
                                    np.array(test_reconstruction_losses),
                                    np.array(test_KL_divergences)), axis=1)

    # Adjusted return statement to include stacked loss arrays
    return train_losses_stacked, test_losses_stacked

def KL_divergence(mu, log_var):
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()


def gaussian_NLL(mu, log_var, x):
    # This prevents negative infinity in the log variance
    const_term = torch.log(torch.tensor(2 * torch.pi, device=log_var.device, dtype=log_var.dtype))
    
    # This computes the Gaussian negative log likelihood with a diagonal covariance matrix
    negative_log_likelihood = 0.5 * (torch.exp(log_var) + (x - mu) ** 2 / torch.exp(log_var) + const_term)
    return torch.sum(negative_log_likelihood)

def loss_fn_ELBO(x_mu, x_log_var, x, mu, log_var, mode = "mse", beta = 1):
    
    if mode =="mse":
        reconstruction_loss = nn.MSELoss()(x_mu, x)
    else:
        reconstruction_loss = gaussian_NLL(x_mu, x_log_var,x)
    KLD = KL_divergence(mu, log_var)
    
    #check nan
    # print(reconstruction_loss, KLD)
    assert not torch.isnan(reconstruction_loss).any()
    assert not torch.isnan(KLD).any()
    return reconstruction_loss, KLD, reconstruction_loss+KLD*beta
