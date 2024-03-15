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

def q3_a(images, vae):
    """
    images: (1000, 32, 32, 3) numpy array in [0, 1], the images to pass through the encoder and decoder of the vae
    vae: a vae model, trained on the relevant dataset

    Returns
    - a numpy array of size (50, 2, 32, 32, 3) of the decoded image in [0, 1] consisting of pairs
      of real and reconstructed images
    - a float that is the scale factor
    """

    """ YOUR CODE HERE """

    #normalize the data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device) #input 0-1, output -1to 1
    vae.eval()
    
    #reconstruct
    normalized_images = (images - 0.5) / 0.5
    normalized_images = normalized_images.transpose(0, 3, 1, 2)
    
    encoded_latents = vae.encode(normalized_images)
    
    autoencoded_images = vae.decode(encoded_latents)
    
    autoencoded_images = autoencoded_images.permute(0,2,3,1).detach().cpu().numpy() / 2 + 0.5
    
    #cat images with autoencoded images in dim 2
    paired_autoencoded_images = np.concatenate([images[:,None], autoencoded_images[:,None]], axis=1)
    
    #calculate scale_factor
    scale_factor = torch.std(encoded_latents.flatten())
    return paired_autoencoded_images[:50], scale_factor.item()

if __name__ == "__main__":
    q3a_save_results(q3_a)
    #test Unet 
    # model = UNet(3)
    # input = torch.randn(1, 3, 32, 32)
    # t = torch.tensor([10])
    # out = model(input, t)