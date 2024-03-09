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
        self.z_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Apply tanh at the output to ensure the output is between -1 and 1
        )
        self.apply(self.init_weights)
    
    def forward(self, x):
        return self.net(x)
    def sample(self, n_samples, device):
        z = torch.randn(n_samples, self.z_dim, device=device)
        return self.forward(z)
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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

criterion = torch.nn.BCELoss()
def g_loss_non_saturating(d_fake_logits):
    targets = torch.ones_like(d_fake_logits)  # Generator wants the discriminator to think the fakes are real.
    return criterion(d_fake_logits, targets)

def d_loss(d_real_logits, d_fake_logits):
    targets_real = torch.ones_like(d_real_logits)
    targets_fake = torch.zeros_like(d_fake_logits)

    loss_real = criterion(d_real_logits, targets_real)
    loss_fake = criterion(d_fake_logits, targets_fake)
    return (loss_real + loss_fake) / 2

def train_gan_q1(generator, discriminator, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn, dataloader, device, epochs=100, debug_mode=False, g_scheduler = None, d_scheduler = None):
    """
    Train a GAN consisting of a generator and discriminator with an option for a quick debug iteration.

    Args:
    - generator: The generator model.
    - discriminator: The discriminator model.
    - g_optimizer: Optimizer for the generator.
    - d_optimizer: Optimizer for the discriminator.
    - g_loss_fn: Loss function for the generator.
    - d_loss_fn: Loss function for the discriminator.
    - dataloader: DataLoader for the real data.
    - device: The device to train on ('cuda' or 'cpu').
    - epochs: Number of epochs to train for.
    - debug_mode: If True, runs a single epoch with a limited number of batches for debugging.

    Returns:
    - A tuple of (generator_losses, discriminator_losses) capturing the loss history.
    """

    generator_losses = []
    discriminator_losses = []

    debug_epochs = 1 if debug_mode else epochs
    debug_batches = 10

    for epoch in tqdm(range(debug_epochs), desc="Epochs"):
        g_epoch_loss = 0
        d_epoch_loss = 0
        batch_count = 0

        for real_data  in dataloader:
            if debug_mode and batch_count >= debug_batches:
                break
            real_data = real_data[0].to(device)
    
            batch_size = real_data.shape[0]
            
            # -----------------
            #  Train Generator
            # -----------------
            g_optimizer.zero_grad()
            # z = torch.randn((batch_size,*generator.shape),device=device, dtype = torch.float32)  # generator.input_size needs to match your generator's input size
            fake_data = generator.sample(batch_size, device = device)
            output = discriminator(fake_data)
            g_loss = g_loss_fn(output)
            g_loss.backward()
            g_optimizer.step()
            if g_scheduler is not None:
                g_scheduler.step()

            # ---------------------
            #  Train Discriminator
            # --------------------
            d_optimizer.zero_grad()
            real_output = discriminator(real_data)
            
            fake_output = discriminator(fake_data.detach())

            d_loss = d_loss_fn(real_output, fake_output)
            d_loss.backward()
            d_optimizer.step()
            if d_scheduler is not None:
                d_scheduler.step()

       
           
            
            d_epoch_loss += d_loss.item()
            g_epoch_loss += g_loss.item()

            batch_count += 1

        generator_losses.append(g_epoch_loss / batch_count)
        discriminator_losses.append(d_epoch_loss / batch_count)

        if epoch == 0:
            samples, samples_interpolate, discriminator_output = evaluate_generator_discriminator(generator, discriminator, device)
        if debug_mode:
            print(f'Debug Mode: Epoch [{epoch+1}/{debug_epochs}], Generator Loss: {g_epoch_loss / batch_count}, Discriminator Loss: {d_epoch_loss / batch_count}')

    return generator_losses, discriminator_losses, samples, samples_interpolate, discriminator_output


def q1_b(train_data):
    """
    train_data: An (20000, 1) numpy array of floats in [-1, 1]

    Returns
    - a (# of training iterations,) numpy array of discriminator losses evaluated every minibatch
    - a numpy array of size (5000,) of samples drawn from your model at epoch #1
    - a numpy array of size (100,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (100,), corresponding to the discriminator output (after sigmoid) 
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
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=256, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    #optimizer
    #Training optimizer
    d_optimizer = optim.Adam(discriminator.parameters(), lr=hyperparams["lr"])
    g_optimizer = optim.Adam(generator.parameters(), lr=hyperparams["g_lr"])
    
    total_steps = hyperparams["num_epochs"] * (len(train_loader))
    lambda_lr = lambda step: 1 - step / total_steps
    d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda_lr)
    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda_lr)

    geneartor_loss, discriminator_loss, samples_ep1, samples_interpolate_ep1, discriminator_output_ep1 = train_gan_q1(
        dataloader=train_loader,
        generator=generator,
        discriminator=discriminator,
        g_loss_fn=g_loss_non_saturating,
        d_loss_fn=d_loss,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g_scheduler=g_scheduler,
        d_scheduler=d_scheduler,
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
   q1_save_results('b', q1_b)