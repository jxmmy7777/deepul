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
import torch
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

#adopted from tutorial : https://www.youtube.com/watch?v=4LktBHGCNfw
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, num_features=64, num_residuals=9):
        super(Generator, self).__init__()
        # Initial convolution block
        self.initial = nn.Sequential(
            nn.Conv2d(input_channel, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList([
            ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
            ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
        ])

        self.res_blocks = nn.Sequential(*[ResidualBlock(num_features * 4) for _ in range(num_residuals)])

        self.up_blocks = nn.ModuleList([
            ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        ])
   
        self.last = nn.Conv2d(num_features, output_channel, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        # Pass through the initial layer
        #Note that we are not sampling from z, but input image for transformation
        x = self.initial(x)
        
        # -----Downsample----
        for block in self.down_blocks:
            x = block(x)
        x = self.res_blocks(x)
        # Upsample
        for block in self.up_blocks:
            x = block(x)
        x = self.last(x)
        return torch.tanh(x)



class Discriminator(nn.Module):
    def __init__(self, img_size, dim):
        super(Discriminator, self).__init__()
        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Conv2d(img_size[2], dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )

        output_size = 8 * dim * int(img_size[0] / 16) * int(img_size[1] / 16)
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        x = self.image_to_features(input_data)
        x = x.view(input_data.size(0), -1)
        return self.features_to_prob(x)
# data class for mnist,c-mnist dataloader, this should

class MNISTData(data.Dataset):
    def __init__(self, minst_data,cmnist_data, transform=None):
        self.minst_data = minst_data
        self.cmnist_data = cmnist_data
        self.transform = transform
        self.length_data_set = max(len(self.minst_data),len(self.cmnist_data))

    def __getitem__(self, index):
        x = self.minst_data[index]
        x0 = self.cmnist_data[index]
        if self.transform:
            augmentations = self.transform(image=x,image0 = x0)
            x = augmentations["image"]
            x0 = augmentations["image0"]
        return x,x0
    def __len__(self):
        return self.length_data_set
def train_gan_q4(disc_gray,disc_color,g_gray,g_color, g_optimizer, d_optimizer, dataloader, device, epochs=100, debug_mode=False, g_scheduler=None, d_scheduler=None):
    generator_losses = []
    discriminator_losses = []
    mse = nn.MSELoss()
    L1 = nn.L1Loss()
    debug_batches = 1
    epochs = 1 if debug_mode else epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        g_epoch_loss, d_epoch_loss, epoch_perceptual_loss, epoch_l2_recon_loss = 0, 0, 0, 0
        batch_count = 0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for idx,(mnist,cmnist) in enumerate(loop):
            if debug_mode and batch_count >= debug_batches:
                break
            mnist = mnist.to(device)
            cmnist = cmnist.to(device)
            
            #---------------Train Discriminator------------------
            #generate fake cmnist
            fake_cmnist = g_color(mnist)
            d_cmnist_fake = disc_color(fake_cmnist.detach())
            d_cmnist_real = disc_color(cmnist)
            d_cmnist_loss_fake = mse(d_cmnist_fake, torch.zeros_like(d_cmnist_fake)) 
            d_cmnist_loss_real = mse(d_cmnist_real, torch.ones_like(d_cmnist_real))
            d_cmnist_loss = (d_cmnist_loss_fake + d_cmnist_loss_real)
            #generate fake mnist
            fake_mnist = g_gray(cmnist)
            d_mnist_fake = disc_gray(fake_mnist.detach())
            d_mnist_real = disc_gray(mnist)
            d_mnist_loss_fake = mse(d_mnist_fake, torch.zeros_like(d_mnist_fake))
            d_mnist_loss_real = mse(d_mnist_real, torch.ones_like(d_mnist_real))
            d_mnist_loss = (d_mnist_loss_fake + d_mnist_loss_real)
            
            d_loss = (d_cmnist_loss + d_mnist_loss)/2
            
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            #---------------Train Geneartor------------------
            #adversarial loss
            fake_cmnist = g_color(mnist)
            fake_mnist = g_gray(cmnist)
            d_fake_cmnist = disc_color(fake_cmnist)
            d_fake_mnist = disc_gray(fake_mnist)
            loss_g_cmnist = mse(d_fake_cmnist, torch.ones_like(d_fake_cmnist))
            loss_g_mnist = mse(d_fake_mnist, torch.ones_like(d_fake_mnist))
            
            #cycle loss
            cycle_mnist = g_gray(fake_cmnist)
            cycle_cmnist = g_color(fake_mnist)
            
            loss_cycle_mnist = L1(cycle_mnist, mnist)
            loss_cycle_cmnist = L1(cycle_cmnist, cmnist)
            
            g_loss = (loss_g_cmnist + loss_g_mnist) + (loss_cycle_mnist + loss_cycle_cmnist) * 10
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            
            # Track losses
            generator_losses.append(g_loss.item())
            discriminator_losses.append(d_loss.item())
           
            if debug_mode:
                break

        print(f'Debug Mode: Epoch [{epoch+1}/{epochs}], Generator Loss: {np.mean(generator_losses)}, Discriminator Loss: {np.mean(discriminator_losses)}')

    return generator_losses, discriminator_losses
def q4(mnist_data, cmnist_data):
    """
    mnist_data: An (60000, 1, 28, 28) numpy array of black and white images with values in [0, 1]
    cmnist_data: An (60000, 3, 28, 28) numpy array of colored images with values in [0, 1]

    Returns
    - a (20, 28, 28, 1) numpy array of real MNIST digits, in [0, 1]
    - a (20, 28, 28, 3) numpy array of translated Colored MNIST digits, in [0, 1]
    - a (20, 28, 28, 1) numpy array of reconstructed MNIST digits, in [0, 1]

    - a (20, 28, 28, 3) numpy array of real Colored MNIST digits, in [0, 1]
    - a (20, 28, 28, 1) numpy array of translated MNIST digits, in [0, 1]
    - a (20, 28, 28, 3) numpy array of reconstructed Colored MNIST digits, in [0, 1]
    """
    """ YOUR CODE HERE """

    num_epochs = 10
    generator_color = Generator(input_channel=1,output_channel=3, num_features=64, num_residuals=3)
    discriminator_gray = Discriminator((28, 28, 1), 64)

 
    generator_gray = Generator(input_channel=3,output_channel=1, num_features=64, num_residuals=3) #take in 3 channels output 1 channel?
    discriminator_color = Discriminator((28, 28, 3), 64)
    
    # Create DataLoader without additional transformations
    # train_tensor = torch.tensor(train_data, dtype = torch.float32)
    # valid_tensor = torch.tensor(val_data, dtype = torch.float32)
    mnist_data = torch.tensor(mnist_data, dtype = torch.float32)
    cmnist_data = torch.tensor(cmnist_data, dtype = torch.float32)
    
    mnist_data  =  mnist_data*2 -1
    cmnist_data =  cmnist_data*2 -1
    
    dataset = MNISTData(mnist_data,cmnist_data)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator_gray = generator_gray.to(device)
    generator_color = generator_color.to(device)
    discriminator_gray = discriminator_gray.to(device)
    discriminator_color = discriminator_color.to(device)

    #optimizer
    #Training optimizer
    d_optimizer = optim.Adam(
        list(discriminator_gray.parameters()) + list(discriminator_color.parameters()), 
        lr=1e-4,
        betas = (0.5, 0.99)
    )
    g_optimizer = optim.Adam(
        list(generator_gray.parameters()) + list(generator_color.parameters()), 
        lr=1e-4,
        betas = (0.5, 0.99),
    )
    
    geneartor_loss, discriminator_loss= train_gan_q4(
        disc_gray = discriminator_gray,
        disc_color = discriminator_color,
        g_gray = generator_gray,
        g_color = generator_color,
        dataloader=train_loader,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        # checkpoint_path=f"homeworks/hw3/results/q1a",
        epochs =num_epochs,
        device=device,
        debug_mode=True,
    )
    
    real_mnist = mnist_data[:20].to(device)
    real_cmnist = cmnist_data[:20].to(device)
    
    #transform mnist
    translated_mnist = generator_color(real_mnist)
    reconstructed_mnist = generator_gray(translated_mnist)
    
    #transform cmnist
    translated_cmnist = generator_gray(real_cmnist)
    reconstructed_cmnist = generator_color(translated_cmnist)
    
    #unormalized backk all
    real_mnist = (real_mnist + 1)/2
    translated_mnist = (translated_mnist + 1)/2
    reconstructed_mnist = (reconstructed_mnist + 1)/2
    real_cmnist = (real_cmnist + 1)/2
    translated_cmnist = (translated_cmnist + 1)/2
    reconstructed_cmnist = (reconstructed_cmnist + 1)/2
    
    
    return (
        real_mnist.permute(0,2,3,1).detach().cpu().numpy(),
        translated_mnist.permute(0,2,3,1).detach().cpu().numpy(),
        reconstructed_mnist.permute(0,2,3,1).detach().cpu().numpy(),
        real_cmnist.permute(0,2,3,1).detach().cpu().numpy(),
        translated_cmnist.permute(0,2,3,1).detach().cpu().numpy(),
        reconstructed_cmnist.permute(0,2,3,1).detach().cpu().numpy()
        
    )


if __name__ == "__main__":

   q4_save_results(q4)
