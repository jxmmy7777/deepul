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
from deepul.hw3_helper import *
import deepul.pytorch_util as ptu
import warnings
warnings.filterwarnings('ignore')

from train_utils import *
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Apply tanh at the output to ensure the output is between -1 and 1
        )
    
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Apply sigmoid at the output to ensure the output is between 0 and 1
        )
    
    def forward(self, x):
        return self.net(x)
    
def q1_a(train_data):
    """
    train_data: An (20000, 1) numpy array of floats in [-1, 1]

    Returns
    - a (# of training iterations,) numpy array of discriminator losses evaluated every minibatch
    - a numpy array of size (5000,) of samples drawn from your model at epoch #1
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid) 
        at each location in the previous array at epoch #1

    - a numpy array of size (5000,) of samples drawn from your model at the end of training
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid) 
        at each location in the previous array at the end of training
    """

    """ YOUR CODE HERE """

    hyperparams = {'lr': 1e-3, 'num_epochs': 500, "g_lr": 1e-4}
   
    generator = Generator(1, 128, 1)
    discriminator = Discriminator(1, 128, 1)

    train_tensor = torch.tensor(train_data, dtype = torch.float32)
    # Create DataLoader without additional transformations
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=1000, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    #optimizer
    #Training optimizer
    d_optimizer = optim.Adam(discriminator.parameters(), lr=hyperparams["lr"])
    g_optimizer = optim.Adam(generator.parameters(), lr=hyperparams["g_lr"])

    g_loss_fn = nn.BCELoss()
    d_loss_fn = nn.BCELoss()
    geneartor_loss, discriminator_loss, samples_ep1, samples_interpolate_ep1, discriminator_output_ep1 = train_gan(
        dataloader=train_loader,
        generator=generator,
        discriminator=discriminator,
        g_loss_fn=g_loss_fn,
        d_loss_fn=d_loss_fn,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        # checkpoint_path=f"homeworks/hw3/results/q1a",
        epochs = hyperparams["num_epochs"],
        device=device,
        debug_mode=False,
    )
    
    samples, samples_interpolate, discriminator_output = evaluate_generator_discriminator(generator, discriminator, device)
    
    
    
    return (
        discriminator_loss,
        samples_ep1,
        samples_interpolate_ep1,
        discriminator_output_ep1,
        samples,
        samples_interpolate,
        discriminator_output
    )


    
if __name__ == "__main__":
   q1_save_results('a', q1_a)