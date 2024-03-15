from deepul.hw4_helper import *
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import deepul.pytorch_util as ptu
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from scipy.stats import norm
from tqdm import trange, tqdm_notebook
import deepul.pytorch_util as ptu
import warnings
warnings.filterwarnings('ignore')

from utils import *
from torch.utils.data import DataLoader, TensorDataset

# MLP 
#MLP with 4 hidden layers and hidden size 64
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=4):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(layers - 1)],
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x, t):
        x = torch.cat([x, t], dim=-1)
        return self.net(x)
#Continous Noise Schedule

def noise_strength(t):
    alpha_t = torch.cos(torch.pi / 2 * t)
    sigma_t = torch.sin(torch.pi / 2 * t)
    return alpha_t, sigma_t
class ContinuousGaussianDiffusion(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # The neural network model f(x,t)
        self.z_dim = 2

    @staticmethod
    def noise_strength(t):
        alpha_t = torch.cos(torch.pi / 2 * t)
        sigma_t = torch.sin(torch.pi / 2 * t)
        return alpha_t, sigma_t

    # 
    @staticmethod
    def DDPM_UPDATE(x, eps_hat, t, tm1):
        alpha_t, sigma_t = noise_strength(t)
        alpha_tm1, sigma_tm1 = noise_strength(tm1)
        
        #𝜂_t
        noise_scale_t =  sigma_tm1/sigma_t * torch.sqrt(1 - (alpha_t/alpha_tm1)**2)
        x_tm1 = alpha_tm1 * (x - sigma_t * eps_hat)/alpha_t + \
                    torch.sqrt(torch.relu(sigma_tm1**2 -  noise_scale_t**2))*eps_hat +\
                    noise_scale_t*torch.randn_like(x)
        return x_tm1
    @torch.no_grad()
    def ddpm_smapler(self, num_steps, num_samples = 2000, device="cuda"):
        ts = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1, device=device)
        x =  torch.randn(num_samples, self.z_dim, device=device)
      
        for i in range(num_steps):
            t = ts[i]
            tm1 = ts[i + 1]
            #t should be same shape as x
            t = t.expand(x.shape[0], 1)
            tm1 = tm1.expand(x.shape[0], 1)
            eps_hat = self.model(x, t)
            x = self.DDPM_UPDATE(x, eps_hat, t, tm1)
        return x

    def forward_diffusion(self, x, t):
        alpha_t, sigma_t = noise_strength(t)
        epsilon = torch.randn_like(x) #uniform sampling
        x_noised = alpha_t * x + sigma_t * epsilon
        return x_noised, epsilon

    def loss_function(self, x_noised, epsilon, t):
        epsilon_hat = self.model(x_noised, t)  #  𝜖̂ =𝑓𝜃(𝑥𝑡,𝑡)
        return F.mse_loss(epsilon, epsilon_hat, reduction= "mean") 

    def loss(self, x):
        t = torch.rand((*x.shape[:-1], 1), device=x.device) #uniform from 0-1
        x_noised, epsilon = self.forward_diffusion(x, t)
        loss = self.loss_function(x_noised, epsilon, t)
        
        return {"loss":loss}



def q1(train_data, test_data):
    """
    train_data: A (100000, 2) numpy array of 2D points
    test_data: A (10000, 2) numpy array of 2D points

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (9, 2000, 2) of samples drawn from your model.
      Draw 2000 samples for each of 9 different number of diffusion sampling steps
      of evenly logarithmically spaced integers 1 to 512
      hint: np.power(2, np.linspace(0, 9, 9)).astype(int)
    """

    """ YOUR CODE HERE """
    #normalize the data
    train_args = {'epochs': 100, 'lr': 1e-3, "scheduler":True}
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    
    mlp = MLP(input_size=3, hidden_size=64, output_size=2)
    
    model = ContinuousGaussianDiffusion(mlp)
    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    # Convert to PyTorch tensors
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_tensor, train_tensor) # Assuming you want to use train_data both as input and target
    test_dataset = TensorDataset(test_tensor, test_tensor) # Same assumption as above

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024)


    train_losses, test_losses = train_epochs(model, train_loader, test_loader, train_args, quiet=False)
    
    #sample from diffusion model
    sample_steps = np.power(2, np.linspace(0, 9, 9)).astype(int)
    samples_list = []
    for i in range(len(sample_steps)):
        samples = model.ddpm_smapler(sample_steps[i])
        samples_list.append(samples.detach().cpu().numpy())
    all_samples = np.array(samples_list)
    #unormalize samples
    all_samples = all_samples * std + mean
    return train_losses["loss"], test_losses["loss"], all_samples

if __name__ == "__main__":
    q1_save_results(q1)