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

from models import *

def generator_loss(discriminator_output, dummy):
    return -discriminator_output.mean()
def discriminator_loss_fn(discriminator_output_real, discriminator_output_fake):
    return -discriminator_output_real.mean() + discriminator_output_fake.mean()

def q2(train_data):
    """
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of WGAN critic train losses evaluated every minibatch
    - a (1000, 32, 32, 3) numpy array of samples from your model in [0, 1]. 
        The first 100 will be displayed, and the rest will be used to calculate the Inception score. 
    """

    """ YOUR CODE HERE """
    hyperparams = {
        "num_epochs":10
        
    }
    generator = Generator_SNGAN(n_filters=128)
    discriminator = Discriminator(n_filters=128)

    train_tensor = torch.tensor(train_data, dtype = torch.float32)
    # Create DataLoader without additional transformations
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    #optimizer
    #Training optimizer
    total_steps = hyperparams["num_epochs"] * (len(train_loader) / 128)

    lambda_lr = lambda step: 1 - step / total_steps

    #2𝑒−4 ,  𝛽1=0 ,  𝛽2=0.9 ,  𝜆=10 ,
    d_optimizer = optim.Adam(discriminator.parameters(), lr = 2e-4, betas=(0, 0.9))
    g_optimizer = optim.Adam(generator.parameters(), lr = 2e-4, betas=(0, 0.9))
    
    d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda_lr)
    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda_lr)

    g_loss_fn = generator_loss
    d_loss_fn = discriminator_loss_fn
    geneartor_loss, discriminator_loss, samples_ep1, samples_interpolate_ep1, discriminator_output_ep1 = train_gan(
        dataloader=train_loader,
        generator=generator,
        discriminator=discriminator,
        g_loss_fn=g_loss_fn,
        d_loss_fn=d_loss_fn,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        g_scheduler=g_scheduler,
        d_scheduler=d_scheduler,
        # checkpoint_path=f"homeworks/hw3/results/q1a",
        epochs = hyperparams["num_epochs"],
        device=device,
        debug_mode=True,
        wgan_gp = True
    )
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
   
    #optimizer
    #Training optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyperparams["lr"])


    return losses, samples

if __name__ == "__main__":
   q2_save_results(q2)