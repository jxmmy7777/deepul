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

from models import DiT

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
        # x = torch.clamp(x, -1, 1)
        alpha_t, sigma_t = noise_strength(t)
        alpha_tm1, sigma_tm1 = noise_strength(tm1)
        
        expand_shape = (-1,) + (1,) * (x.dim() - 1)  # Creates a shape like [-1, 1, 1, 1] for 4D or [-1, 1] for 2D
        alpha_t = alpha_t.view(expand_shape)
        sigma_t = sigma_t.view(expand_shape)
        alpha_tm1 = alpha_tm1.view(expand_shape)
        sigma_tm1 = sigma_tm1.view(expand_shape)
        #𝜂_t
        noise_scale_t =  sigma_tm1/sigma_t * torch.sqrt(1 - (alpha_t/alpha_tm1)**2)
        x_hat =  (x - sigma_t * eps_hat)/alpha_t
        # x_hat = torch.clamp(x_hat, -1, 1)
        x_tm1 = alpha_tm1 * x_hat  + \
                    torch.sqrt(torch.relu(sigma_tm1**2 -  noise_scale_t**2))*eps_hat +\
                    noise_scale_t*torch.randn_like(x)
        # x_tm1 = torch.clamp(x_tm1, -1, 1)
        return x_tm1
    @torch.no_grad()
    def ddpm_smapler(self, num_steps, num_samples = 2000, device="cuda"):
        ts = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1, device=device)
        x =  torch.randn((num_samples,*self.model.input_shape) , device=device)
        # x  = torch.clamp(x, -1,1)
        for i in range(num_steps):
            t = ts[i]
            tm1 = ts[i + 1]
            #t should be same shape as x
            t = t.expand(x.shape[0])
            tm1 = tm1.expand(x.shape[0])
            eps_hat = self.model(x, t)
            x = self.DDPM_UPDATE(x, eps_hat, t, tm1)
        return x

    @torch.no_grad()
    def ddpm_smapler_class(self, num_steps,y, num_samples = 2000, device="cuda", cfg=False, w = 1.0):
        ts = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1, device=device)
        x =  torch.randn((num_samples,*self.model.input_shape) , device=device)
        y = torch.tensor([y], device=device)
        for i in range(num_steps):
            t = ts[i]
            tm1 = ts[i + 1]
            #t should be same shape as x
            t = t.expand(x.shape[0])
            tm1 = tm1.expand(x.shape[0])
            y = y.expand(x.shape[0])
            
            if cfg:
                eps_hat = self.model.forward_cfg(x, t, y, w=w)
            else:
                eps_hat = self.model(x, t, y)
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

    def loss_function(self, x_noised, epsilon, t, y = None):
        epsilon_hat = self.model(x_noised, t, y)  #  𝜖̂ =𝑓𝜃(𝑥𝑡,𝑡)
        return F.mse_loss(epsilon, epsilon_hat, reduction= "mean") 

    def loss(self, x, y=None):
        t = torch.rand((x.shape[0]), device=x.device) #uniform from 0-1, but should be same accross batch
        #if t is not in the same shape as x, need to braodcast
       
        x_noised, epsilon = self.forward_diffusion(x, t)
        loss = self.loss_function(x_noised, epsilon, t, y)
        
        return {"loss":loss}
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5
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

    # return train_losses, test_losses, samples
    #normalize the data
  
    train_args = {'epochs': 60, 'lr': 1e-3}
    # ---------------------data--------------------------------
    train_data = normalize_to_neg_one_to_one(train_data).transpose(0, 3, 1, 2)
    test_data =  normalize_to_neg_one_to_one(test_data).transpose(0, 3, 1, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --------------------use vae to transform---------------------
    vae.to(device) #input 0-1, output -1to 1
    vae.eval()
    scale_factor = 1.2630
    # scale_factor = 1/0.18#1.2630
    with torch.no_grad():
        train_latents = vae.encode(train_data)/ scale_factor #4, 8,8 
        test_latents = vae.encode(test_data)/ scale_factor
    #calculate scale_factor
    # scale_factor = torch.std(train_latents.flatten())
    # train_latents = train_latents / scale_factor
    # test_latents = test_latents / scale_factor
    print(f"scale_fcator{scale_factor}")
    train_labels = torch.tensor(train_labels,dtype=torch.long)
    test_labels = torch.tensor(test_labels,dtype=torch.long)
    # Create TensorDatasets
    train_dataset = TensorDataset(train_latents, train_labels) # Assuming you want to use train_data both as input and target
    test_dataset = TensorDataset(test_latents, test_labels) # Same assumption as above

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)

    total_steps = len(train_loader) * train_args["epochs"]
    train_args["scheduler"] = {"Total_steps":total_steps, "Warmup_steps":100}
    #TODO pass scheduler config to train_args
    num_classes = 10
    # --------------models----------------
    DiT_config = {
        "input_shape": (4,8,8), 
        "patch_size": 2, 
        "hidden_size": 512, 
        "num_heads": 8, 
        "num_layers": 12, 
        "num_classes": num_classes, 
        "cfg_dropout_prob":0.1
    }
    Diffusion_transformer = DiT(**DiT_config)
    Diffusion_transformer.input_shape = (4,8,8)
    
    # from DiT_reference import DiT
    # DiT_config = {
    #     "input_size": 8,
    #     "in_channels":4, 
    #     "patch_size": 2, 
    #     "hidden_size": 512, 
    #     "num_heads": 8, 
    #     "depth": 12, 
    #     "num_classes": num_classes, 
    #     "learn_sigma": False
    #     # "cfg_dropout_prob":0.1
    # }
    
    # Diffusion_transformer = DiT(**DiT_config)
    # Diffusion_transformer.input_shape = (4,8,8)
    
    model = ContinuousGaussianDiffusion(Diffusion_transformer)
    
    # model.load_state_dict(torch.load("checkpoints/q3b_diffusiontransformer_change_atten.pth")["model_state_dict"])
    # train_losses = {}
    # test_losses = {}
    # train_losses["loss"] = [0]
    # test_losses["loss"] = [0]
    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, test_losses = train_epochs(model, train_loader, test_loader, train_args, quiet=False, 
                                             checkpoint=f"checkpoints/q3b_diffusiontransformer_change_atten.pth")
    
    #sample from diffusion model
    model.eval()
    num_classes_list = np.arange(num_classes)
    samples_list = []
    for i in range(len(num_classes_list)):
        samples = model.ddpm_smapler_class(512, i, num_samples=10)
        samples_list.append(samples)
    all_samples = torch.stack(samples_list)
    # vae decodoing 
    all_samples = all_samples.reshape(10*10,4,8,8) * scale_factor #10,10, 4,8,8 -> 100,4,8,8
    all_samples_decoded = vae.decode(all_samples).reshape(10,10,3,32,32) #10,10, 3,32,32
    all_samples_decoded = unnormalize_to_zero_to_one(all_samples_decoded)
    return train_losses["loss"], test_losses["loss"], all_samples_decoded.permute(0,1,3,4,2).detach().cpu().numpy()

def q3_c(vae):
    """
    vae: a pretrained vae

    Returns
    - a numpy array of size (4, 10, 10, 32, 32, 3) of samples in [0, 1] drawn from your model.
      The array represents a 4 x 10 x 10 grid of generated samples - 4 10 x 10 grid of samples
      with 4 different CFG values of w = {1.0, 3.0, 5.0, 7.5}. Each row of the 10 x 10 grid
      should contain samples of a different class. Use 512 diffusion sampling timesteps.
    """

    """ YOUR CODE HERE """
    scale_factor = 1.2630
    num_classes = 10
    # --------------models----------------
    DiT_config = {
        "input_shape": (4,8,8), 
        "patch_size": 2, 
        "hidden_size": 512, 
        "num_heads": 8, 
        "num_layers": 12, 
        "num_classes": num_classes, 
        "cfg_dropout_prob":0.1
    }
    Diffusion_transformer = DiT(**DiT_config)
    Diffusion_transformer.input_shape = (4,8,8)
    
    model = ContinuousGaussianDiffusion(Diffusion_transformer)
    
    model.load_state_dict(torch.load("checkpoints/q3b_diffusiontransformer_change_atten.pth")["model_state_dict"])
    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
     #sample from diffusion model
    model.eval()
    num_classes_list = np.arange(num_classes)
    samples_list = []
    w_list = {1.0, 3.0, 5.0, 7.5}
    for w in w_list:
        for i in range(len(num_classes_list)):
            samples = model.ddpm_smapler_class(512, i, num_samples=10, cfg=True, w=w)
            samples_list.append(samples)
            
    all_samples = torch.stack(samples_list)
    # vae decodoing 
    all_samples = all_samples.reshape(4*10*10,4,8,8) * scale_factor #10,10, 4,8,8 -> 100,4,8,8
    all_samples_decoded = vae.decode(all_samples).reshape(4,10,10,3,32,32) #10,10, 3,32,32
    all_samples_decoded = unnormalize_to_zero_to_one(all_samples_decoded)

    return all_samples_decoded.permute(0,1,2,4,5,3).detach().cpu().numpy()


if __name__ == "__main__":
    # q3b_save_results(q3_b)
    q3c_save_results(q3_c)
    #test Unet 
    # model = UNet(3)
    # input = torch.randn(1, 3, 32, 32)
    # t = torch.tensor([10])
    # out = model(input, t)