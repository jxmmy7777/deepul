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

from DiT import DiT

def noise_strength(t):
    alpha_t = torch.cos(torch.pi / 2 * t)
    sigma_t = torch.sin(torch.pi / 2 * t)
    return alpha_t, sigma_t

def timestep_embedding(timesteps, dim, max_period=10000):
    half_dim = dim // 2
    device = timesteps.device  
    freqs = torch.exp(-torch.log(torch.tensor(max_period, device=device, dtype=torch.float32)) * torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim)
    
    # Ensure timesteps is a float tensor for multiplication
    timesteps = timesteps.float().unsqueeze(-1)
    args = timesteps * freqs
    
    # Calculate sin and cos components
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # If dim is odd, pad with a column of zeros
    if dim % 2:
        zero_pad = torch.zeros(embedding.shape[0], 1, device=device, dtype=torch.float32)
        embedding = torch.cat([embedding, zero_pad], dim=-1)
    
    return embedding #(B,D)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, num_groups=8):
        super(ResidualBlock,self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        # Define the linear layer for temb
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        # Optional: adjust channels if in_channels != out_channels
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
    def forward(self, x, temb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        

        temb = self.temb_proj(temb)
        h += temb[:, :, None, None] # h is BxDxHxW, temb is BxDx1x1

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        x = self.proj(x)
        return x + h

def Downsample(in_channels):
    return nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

def Upsample(in_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_channels, in_channels, 3, padding=1),
    )

# refhttps://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L111
class UNet(nn.Module):
    def __init__(self, in_channels, hidden_dims = [64, 128, 256, 512] , blocks_per_dim = 2):
        super(UNet, self).__init__()  # Correctly initialize the superclass.
        self.temb_channels = hidden_dims[0] * 4
        
        self.input_shape = (in_channels, 32, 32)
        
        self.first_hidden_dim = hidden_dims[0]
        self.time_emb_mlp = nn.Sequential(
            nn.Linear(hidden_dims[0],  self.temb_channels),
            nn.SiLU(),
            nn.Linear(self.temb_channels,  self.temb_channels)
        )
        
        #blocks
        self.init_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList()
        prev_ch = hidden_dims[0]
        down_block_chans = [prev_ch]
        in_out = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        
        
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            self.downs.append(
                nn.ModuleList([
                    ResidualBlock(dim_in, dim_out, self.temb_channels),
                    ResidualBlock(dim_out, dim_out, self.temb_channels),
                    Downsample(dim_out) if not is_last else nn.Identity()
                ])
            )
        
        prev_ch = hidden_dims[-1]
        # Middle blocks
        self.middle_block_1 = ResidualBlock(prev_ch, prev_ch, self.temb_channels)
        self.middle_block_2 = ResidualBlock(prev_ch, prev_ch, self.temb_channels)
        
        # Upsample path
        self.ups = nn.ModuleList()
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            
            input_dim = (dim_in + dim_out)*2 if ind > 0 else dim_out * 2
            self.ups.append(nn.ModuleList([
                ResidualBlock( input_dim, dim_out, self.temb_channels),
                ResidualBlock(dim_out, dim_out, self.temb_channels),
                Upsample(dim_out) if not is_last else nn.Identity()
                
            ]))

        prev_ch = dim_out
        self.final_norm = nn.GroupNorm(num_groups=8, num_channels=prev_ch)
        self.final_conv = nn.Conv2d(prev_ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        
        temb = timestep_embedding(t, self.first_hidden_dim)
        temb = self.time_emb_mlp(temb)
        hs = []
        h = self.init_conv(x)
        hs.append(h)
        
        # Downsampling
        for resnet, resnet2, down in self.downs:
            h = resnet(h, temb)
            h = resnet2(h, temb)
            hs.append(h)
            h = down(h)
        # Middle
        h = self.middle_block_1(h, temb)
        h = self.middle_block_2(h, temb)
        # Upsampling
        #hs [torch.Size([1, 64, 32, 32]), torch.Size([1, 128, 32, 32]), torch.Size([1, 256, 16, 16]), torch.Size([1, 512, 8, 8])]
        for resnet, resnet2, up in self.ups:
            h = torch.cat([h, hs.pop()], dim=1) #2*hidden_dim , next hidden_dim+prev_hiddem
            h = resnet(h, temb)
            h = resnet2(h, temb)
            h = up(h)
            
            
       
        h = self.final_norm(h)
        h = F.silu(h)
        out = self.final_conv(h)
        return out
 

class ContinuousGaussianDiffusion(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # The neural network model f(x,t)

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
        
        expand_shape = (-1,) + (1,) * (x.dim() - 1)  # Creates a shape like [-1, 1, 1, 1] for 4D or [-1, 1] for 2D
        alpha_t = alpha_t.view(expand_shape)
        sigma_t = sigma_t.view(expand_shape)
        alpha_tm1 = alpha_tm1.view(expand_shape)
        sigma_tm1 = sigma_tm1.view(expand_shape)
        #𝜂_t
        noise_scale_t =  sigma_tm1/sigma_t * torch.sqrt(1 - (alpha_t/alpha_tm1)**2)
        x_tm1 = alpha_tm1 * (x - sigma_t * eps_hat)/alpha_t + \
                    torch.sqrt(torch.relu(sigma_tm1**2 -  noise_scale_t**2))*eps_hat +\
                    noise_scale_t*torch.randn_like(x)
        #clip x_tm1 to -1 to 1
        x_tm1 = torch.clamp(x_tm1, -1, 1)
        return x_tm1
    @torch.no_grad()
    def ddpm_smapler(self, num_steps, num_samples = 2000, device="cuda"):
        ts = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1, device=device)
        x =  torch.randn((num_samples,*self.model.input_shape) , device=device)
      
        for i in range(num_steps):
            t = ts[i]
            tm1 = ts[i + 1]
            #t should be same shape as x
            t = t.expand(x.shape[0])
            tm1 = tm1.expand(x.shape[0])
            eps_hat = self.model(x, t)
            x = self.DDPM_UPDATE(x, eps_hat, t, tm1)
        return x

    def forward_diffusion(self, x, t):
        alpha_t, sigma_t = noise_strength(t)
        epsilon = torch.randn_like(x) #uniform sampling 
        #alpha_t,sigma_t is shape (B), should expand to same shape as x?
        if len(x.shape) == 4:
            alpha_t = alpha_t[:,None,None,None]
            sigma_t = sigma_t[:,None,None,None]
        x_noised = alpha_t * x + sigma_t * epsilon
        return x_noised, epsilon

    def loss_function(self, x_noised, epsilon, t):
        epsilon_hat = self.model(x_noised, t)  #  𝜖̂ =𝑓𝜃(𝑥𝑡,𝑡)
        return F.mse_loss(epsilon, epsilon_hat, reduction= "mean") 

    def loss(self, x):
        t = torch.rand((x.shape[0]), device=x.device) #uniform from 0-1, but should be same accross batch
        #if t is not in the same shape as x, need to braodcast
       
        x_noised, epsilon = self.forward_diffusion(x, t)
        loss = self.loss_function(x_noised, epsilon, t)
        
        return {"loss":loss}

def q3_b(train_data, train_labels, test_data, test_labels, vae):
    """
    train_data: A (50000, 32, 32, 3) numpy array of images in [0, 1]
    train_labels: A (50000,) numpy array of class labels
    test_data: A (10000, 32, 32, 3) numpy array of images in [0, 1]
    test_labels: A (10000,) numpy array of class labels
    vae: a pretrained VAE

    Returns
    - a (# of training iterations,) numpy array of train losses evaluated every minibatch
    - a (# of num_epochs + 1,) numpy array of test losses evaluated at the start of training and the end of every epoch
    - a numpy array of size (10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 10 x 10 grid of generated samples. Each row represents 10 samples generated
      for a specific class (i.e. row 0 is class 0, row 1 class 1, ...). Use 512 diffusion timesteps
    """

    """ YOUR CODE HERE """

    return train_losses, test_losses, samples
    #normalize the data
    train_args = {'epochs': 60, 'lr': 1e-3}
    DiT_config = {
        "input_shape": (4,8,8), 
        "patch_size": 2, 
        "hidden_size": 512, 
        "num_heads": 8, 
        "num_layers": 12, 
        "num_classes": 10, 
        "cfg_dropout_prob":0.1
    }
    DiT = DiT(**DiT_config)
    
    model = ContinuousGaussianDiffusion(Unet)
    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_data = (train_data - 0.5)*2
    test_data = (test_data - 0.5)*2
    # Convert to PyTorch tensors
    train_tensor = torch.tensor(train_data, dtype=torch.float32).permute(0, 3, 1, 2)
    test_tensor = torch.tensor(test_data, dtype=torch.float32).permute(0, 3, 1, 2)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_tensor, train_tensor) # Assuming you want to use train_data both as input and target
    test_dataset = TensorDataset(test_tensor, test_tensor) # Same assumption as above

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)

    total_steps = len(train_loader) * train_args["epochs"]
    train_args["scheduler"] = {"Total_steps":total_steps, "Warmup_steps":100}
    #TODO pass scheduler config to train_args

    train_losses, test_losses = train_epochs(model, train_loader, test_loader, train_args, quiet=False, checkpoint=f"q2")
    
    #sample from diffusion model
    sample_steps = np.power(2, np.linspace(0, 9, 10)).astype(int)
    samples_list = []
    for i in range(len(sample_steps)):
        samples = model.ddpm_smapler(sample_steps[i], num_samples=10).permute(0, 2, 3, 1)
        samples_list.append(samples.detach().cpu().numpy())
    all_samples = np.array(samples_list)
    #unormalize samples
    all_samples = all_samples * 0.5 + 0.5
    return train_losses["loss"], test_losses["loss"], all_samples.reshape(10, 10, 32, 32, 3)

if __name__ == "__main__":
    q2_save_results(q2)
    #test Unet 
    # model = UNet(3)
    # input = torch.randn(1, 3, 32, 32)
    # t = torch.tensor([10])
    # out = model(input, t)