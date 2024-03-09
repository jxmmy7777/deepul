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

from models import *


def train_gan_q2(generator, discriminator, g_optimizer, d_optimizer, dataloader, device, epochs=100, debug_mode=False, n_critic = 5, g_scheduler = None, d_scheduler = None,checkpoint_path=None):
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
    gp_losses = []

    debug_epochs = 1 if debug_mode else epochs
    debug_batches = 10
    batch_count = 0

    for epoch in tqdm(range(debug_epochs), desc="Epochs"):
        g_epoch_loss = 0
        d_epoch_loss = 0
        
        
        for real_data  in dataloader:
            if debug_mode and batch_count >= debug_batches:
                break
            real_data = real_data[0].to(device)
    
            batch_size = real_data.shape[0]
            # train discirminator
            d_optimizer.zero_grad()
            fake_data = generator.sample(batch_size, device = device)
            real_output = discriminator(real_data)
            fake_output = discriminator(fake_data.detach())
            
            
            dis_loss = fake_output.mean()-real_output.mean()
            gp_loss = gradient_penalty(real_data, fake_data, discriminator) 
            d_loss = dis_loss + gp_loss * 10
        
            d_loss.backward()
            d_optimizer.step()
                
            discriminator_losses.append(d_loss.item())
            gp_losses.append(gp_loss.item())

            if d_scheduler is not None:
                d_scheduler.step()
            # train_geneartor
            if batch_count % n_critic == 0:
                g_optimizer.zero_grad()
                # z = torch.randn((batch_size,*generator.shape),device=device, dtype = torch.float32)  # generator.input_size needs to match your generator's input size
                fake_data = generator.sample(batch_size, device = device)
                output = discriminator(fake_data)
                g_loss = -(output).mean()
                g_loss.backward()
                g_optimizer.step()
                if g_scheduler is not None:
                    g_scheduler.step()
                generator_losses.append(g_loss.item())
            
            batch_count += 1

        print(f"Epoch [{epoch+1}/{epochs}], Generator Loss: {np.mean(generator_losses)}, Discriminator Loss: {np.mean(discriminator_losses)}, GP: {np.mean(gp_losses)}")

        if checkpoint_path is not None and batch_count % 500 == 0:
            torch.save(generator.state_dict(), f"{checkpoint_path}_{epoch}_generator.pth")
            torch.save(discriminator.state_dict(), f"{checkpoint_path}_{epoch}_discriminator.pth")
        
    return generator_losses, discriminator_losses, gp_losses
def q2(train_data):
    """
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of WGAN critic train losses evaluated every minibatch
    - a (1000, 32, 32, 3) numpy array of samples from your model in [0, 1]. 
        The first 100 will be displayed, and the rest will be used to calculate the Inception score. 
    """

    """ YOUR CODE HERE """
    #set different seed
    torch.manual_seed(1)
    np.random.seed(1)
    

    generator = Generator_SNGAN(n_filters=128)
    discriminator = Discriminator(n_filters=128)

    train_tensor = torch.tensor(train_data, dtype = torch.float32)
    train_tensor = (train_tensor *2) -1 #nomralized to -1 1
    # Create DataLoader without additional transformations
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=256, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # hyperparmas
    
    n_critic = 5
    total_steps = 25000
    num_epochs = total_steps // len(train_loader)
    
    # Define the step threshold at which you start annealing the learning rate
    d_lambda_lr = lambda step: 1 - step / (total_steps)
    g_lambda_lr = lambda step: 1 - step / (total_steps)

    # Apply these lambda functions to your schedulers

    #2𝑒−4 ,  𝛽1=0 ,  𝛽2=0.9 ,  𝜆=10 ,
    d_optimizer = optim.Adam(discriminator.parameters(), lr = 2e-4, betas=(0.0, 0.9))
    g_optimizer = optim.Adam(generator.parameters(), lr = 2e-5, betas=(0.0, 0.9))
    

    d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=d_lambda_lr)
    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=g_lambda_lr)


    generator_losses, discriminator_loss, gradient_penalty = train_gan_q2(
        dataloader=train_loader,
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g_scheduler=g_scheduler,
        d_scheduler=d_scheduler,
        checkpoint_path=f"hw3_q2",
        epochs = num_epochs,
        device=device,
        debug_mode=False,
        n_critic = n_critic
    )
    
    samples = generator.sample(1000, device = device).permute(0, 2, 3, 1).cpu().detach().numpy()
    #unormallize
    samples = samples/2 + 0.5
    #plot the losses curve
    # Creating a figure for subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column

    # Plotting Generator Loss on the first subplot
    axes[0].plot(generator_losses, label="Generator Loss", color='blue')
    axes[0].set_title("Generator Loss")
    axes[0].set_xlabel("training iter")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plotting Gradient Penalty on the second subplot
    axes[1].plot(gradient_penalty, label="Gradient Penalty", color='green')
    axes[1].set_title("Gradient Penalty")
    axes[1].set_xlabel("training iter")
    axes[1].set_ylabel("Value")
    axes[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("WGAN-GP_Losses_and_Gradient_Penalty.png")
    
    #Fréchet inception distance (bonus, 5pts)
    
    return discriminator_loss, samples

if __name__ == "__main__":
   q2_save_results(q2)