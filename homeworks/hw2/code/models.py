import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv2d, Linear, Flatten, ConvTranspose2d
from torch.nn import LayerNorm  # Ensure this is correctly imported

from abc import abstractmethod
from loss_utils import *
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        # log_var = torch.tanh(log_var)
        # # Scale to [-10, 10]
        # log_var = log_var * 10
        return mu, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_var = nn.Linear(hidden_dim, output_dim)
        self.latent_dim = latent_dim
    def forward(self, z):
        h = self.fc1(z)
        h = self.relu(h)
        h = self.fc2(h)
        h = self.relu(h)
        mean = self.fc_mean(h)
        log_var = self.fc_var(h)
        # log_var = torch.tanh(log_var)
        # # Scale to [-10, 10]
        # log_var = log_var * 10
        return mean, log_var


class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()
        
    def encode(self, x):
        raise NotImplementedError
    
    def decode(self, z):
        raise NotImplementedError
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def loss(self, *inputs, **kwargs):
        # Custom loss logic specific to the model
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, x):
        pass
    
    @torch.no_grad()
    def sample_reconstruct(self, x, device='cpu'):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
       
        return self.sample_without_noise(z, device=device)
    
    @torch.no_grad()
    def sample_with_noise(self,z=None, size=100, device='cpu'):
        if z is None:
            z = torch.randn(size, * self.z_dim, device=device)
        mu, log_var = self.decode(z)  # Ensure your Decoder outputs both
        x_hat = self.reparameterize(mu, log_var)
        return x_hat

    @torch.no_grad()
    def sample_without_noise(self, z=None, size=100, device='cpu'):
        if z is None:
            z = torch.randn(size, *self.z_dim, device=device)
        mu, _ = self.decode(z)  # Use only mu from your Decoder's output
        return mu
    
    @torch.no_grad()
    def sample_interpolate(self, x,x1, interpolate_pt = 10, device='cpu'):
        assert x.shape ==x1.shape
        mu, log_var_1 = self.encode(x)
        mu_2, log_var_2 = self.encode(x1)
        
        z1 = self.reparameterize(mu, log_var_1)
        z2 = self.reparameterize(mu_2, log_var_2)
        z_interpolate = self.interpolate(z1[:,None], z2[:,None])  # Linear interpolation in extra dim=1

        # Decode the interpo
    
        return self.sample_without_noise(z_interpolate.view(-1,*self.z_dim), size=interpolate_pt * interpolate_pt, device=device)
    @staticmethod
    def interpolate(z_start, z_end, size=10):
        weights = torch.linspace(0, 1, steps=size).unsqueeze(0).unsqueeze(-1).to(z_start.device)
        z_interp = (1 - weights) * z_start + weights * z_end
        return z_interp

      
class VAE(BaseVAE):
    def __init__(self, Encoder, Decoder, loss_mode = "mse"):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.loss_mode = loss_mode
        
        self.z_dim = (Decoder.latent_dim)
    
    def encode(self, x):
        return self.Encoder(x)
    
    def decode(self, z):
        return self.Decoder(z) 
    

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_mean, x_log_var = self.Decoder(z)
        
        return  x_mean, x_log_var, mu, log_var
    
    def loss(self, x_mean, x_log_var, mu, log_var, x, beta = 1):
        if self.loss_mode =="gll":
            reconstruction_loss = gaussian_NLL(x_mean, x_log_var, x)
        elif self.loss_mode =="mse":
            reconstruction_loss =  F.mse_loss(x_mean, x, reduction='none').sum(dim=(1,2,3)).mean()
        KLD = KL_divergence(mu, log_var)
        
        total_loss = reconstruction_loss + KLD*beta
        
        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "KLD": KLD,
            "loss": total_loss
        }
        return loss_dict
    

# -----------------------------------q2a-------------------------------
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), # 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), # 8 x 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), # 4 x 4
            nn.ReLU(),
            nn.Flatten(), # 16
            # nn.Linear(4 * 4 * 256, 2 * latent_dim)
        )
        
        self.fc_mu = nn.Linear(4 * 4 * 256,  latent_dim)
        self.fc_var = nn.Linear(4 * 4 * 256,  latent_dim)
    def forward(self, x):
        #
        output = self.net(x)
        # decopule to mu and log_var
        mu = self.fc_mu(output)
        log_var = self.fc_var(output)
        # Scale to [-10, 10]
        # log_var = log_var * 10
        # log_var = torch.tanh(log_var)
        # # Scale to [-10, 10]
        # log_var = log_var * 10
        return mu, log_var
    
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 4 * 4 * 128)  # Prepare for reshaping to a 4x4 feature map
        self.conv_transpose= nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # Output: 8 x 8
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 16 x 16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32 x 32
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)  # Output: 32 x 32 (with 3 channels)
        )
    def forward(self, x):
        #
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)  # Reshape to (Batch, Channels, Height, Width)
        # decopule to mu and log_var
        x = self.conv_transpose(x)
        # mu = torch.tanh(x)
        mu = x
        log_var = torch.zeros_like(mu)
        # Scale to [-10, 10]
        # log_var = torch.tanh(log_var)
        # # Scale to [-10, 10]
        # log_var = log_var * 10
        # log_var = log_var * 10
        return mu, log_var
    
# -----------------------------------q2b-------------------------------
class PermuteLayerNorm(nn.Module):
    def __init__(self, normalized_shape, permute_to=(0, 2, 3, 1), permute_back=(0, 3, 1, 2)):
        """
        A custom layer that permutes dimensions, applies layer normalization, and then permutes back.
        :param normalized_shape: The C dimension to normalize over after permutation.
        :param permute_to: The desired order before normalization (making C the last dimension).
        :param permute_back: The order to revert to after normalization (restoring original order).
        """
        super(PermuteLayerNorm, self).__init__()
        self.permute_to = permute_to
        self.permute_back = permute_back
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        # Permute x to move the channel dimension to the last position
        x_permuted = x.permute(self.permute_to)
        # Apply layer normalization
        normalized = self.norm(x_permuted)
        # Permute back to the original order
        x_normalized = normalized.permute(self.permute_back)
        return x_normalized
class ResidualCorrection(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(ResidualCorrection, self).__init__()
        self.fc_residual = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim) 
        )
        
    def forward(self, z1):
        residual = self.fc_residual(z1)
     
        corrected_z2_mu = z1 + residual
        return corrected_z2_mu
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc_residual = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim ,3,1,1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim ,1,1,0, bias=False)
        )
        
    def forward(self, x):
        residual = self.fc_residual(x)
  
        return x + residual
class HVAE(BaseVAE):
    def __init__(
            self,
            input_channels = 3,
            latent_dim_1 = 12,
            latent_dim_2 = 12,
            latent_img_dim = 2):
        super(HVAE, self).__init__()
        self.latent_dim_1 = latent_dim_1
        self.latent_dim_2 = latent_dim_2
        
        self.Encoder_z1 = nn.Sequential(
            nn.Conv2d(3 , 32, 3, padding=1), # [32, 32, 32]
            PermuteLayerNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # [64, 16, 16]
            PermuteLayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), # [64, 8, 8]
            PermuteLayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), # [64, 4, 4]
            PermuteLayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), # [64, 2, 2]
            PermuteLayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 12*2, 3, padding=1), # [12*2, 2, 2]
        )
        
        self.Encoder_z2 = nn.Sequential(
            nn.Conv2d(3 + 12, 32, 3, padding=1), # [32, 32, 32]
            PermuteLayerNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # [64, 16, 16]
            PermuteLayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), # [64, 8, 8]
            PermuteLayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), # [64, 4, 4]
            PermuteLayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), # [64, 2, 2]
            PermuteLayerNorm(64),
            nn.ReLU(),
            nn.Conv2d(64, 12*2, 3, padding=1), # [12*2, 2, 2]
        )
        
        self.prior_z_2_mu = ResidualCorrection(
            latent_dim_1*latent_img_dim*latent_img_dim,
            latent_dim_1*latent_img_dim*latent_img_dim
            )
        
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(12, 64, 3, padding=1), # [64, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), # [64, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), # [64, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), # [64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # [32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1), # [3, 32, 32]
            nn.Tanh()
        )
        
        self.z_dim = (latent_dim_2, latent_img_dim, latent_img_dim)
        
    def encode_z1(self, x):
        encodede_x = self.Encoder_z1(x) #can input zeros? of z1
        z1_mu, z1_log_var = encodede_x[:, :self.latent_dim_1], encodede_x[:, self.latent_dim_1:]
        return z1_mu, z1_log_var
    
    def encode_z2(self, x, z1):
        #q (z_2 | x, z_1)  residual norm
        #upsampled z1
        _,_, height_x, width_x = x.shape
        z1_upsampled = F.interpolate(z1, size=(height_x, width_x), mode='nearest')
        
        z2_residual_mu, z2_residual_logstd = self.Encoder_z2(torch.cat([x, z1_upsampled], dim=1)).chunk(2, dim=1)
        
        return z2_residual_mu, z2_residual_logstd
    
    def encode(self, x):
        z1_mu, z1_log_var = self.encode_z1(x)
        z1 = self.reparameterize(z1_mu, z1_log_var)
        #q (z_2 | x, z_1)  residual norm
        z2_residual_mu, z2_residual_logstd = self.encode_z2(x, z1)
        #
        # prior p(z_2 | z_1)
        z1_flattened = z1.contiguous().view(-1, self.latent_dim_1 * 2 * 2)
        z2_mu_prior = self.prior_z_2_mu(z1_flattened).view(-1, self.latent_dim_2, 2, 2)
        z2_log_var_prior = torch.zeros_like(z2_mu_prior)
        
        z2_mu = z2_residual_mu + z2_mu_prior
        z2_log_var = z2_residual_logstd + z2_log_var_prior
        
        return z1_mu, z1_log_var, z2_mu, z2_log_var, z2_residual_mu, z2_residual_logstd, z1
    
    def decode(self, z2):
        return self.Decoder(z2), None
    
    def forward(self, x):
        z1_mu, z1_log_var, z2_mu, z2_log_var, z2_residual_mu, z2_residual_logvar, z1 = self.encode(x)
        z2 = self.reparameterize(z2_mu, z2_log_var)
        #upscale z2 to x?
        
        reconstruct, _ = self.decode(z2)
        
        return reconstruct, z1_mu, z1_log_var, z2_residual_mu, z2_residual_logvar, z1, z2
        
    def loss(self, x_mean, z1_mu, z1_log_var, z2_residual_mu, z2_residual_logvar, z1, z2, x, beta = 1):
        
        
        z2_residual_logstd = 0.5 * z2_residual_logvar
        reconstruction_loss =  F.mse_loss(x_mean, x, reduction='none').sum(dim=(1,2,3)).mean()
        kl_z1 = -0.5 * (1 + z1_log_var - z1_mu.pow(2) - z1_log_var.exp())
        kl_z2 = -z2_residual_logstd - 0.5 + (torch.exp(2 * z2_residual_logstd) + z2_residual_mu ** 2) * 0.5
        
        KLD = (kl_z1 + kl_z2).sum(dim= (1,2,3)).mean()
        
        total_loss = reconstruction_loss + KLD*beta
        
        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "KLD": KLD,
            "loss": total_loss
        }
        return loss_dict
        
    @torch.no_grad()
    def sample(self, sample_size, device):
        pass
    
    @torch.no_grad()
    def reconstruct(self, x):
        return self.forward(x)[0]
    
    @torch.no_grad()
    def sample_without_noise(self, z1=None, size=100, device='cpu'):
        #Note! You should sample on Z1!
        if z1 is None:
            z1 = torch.randn(size, *self.z_dim, device=device)
        
        # x = torch.zeros(size, 3, 32, 32, device=device)
        # z2_residual_mu, z2_residual_logstd = self.encode_z2(x, z1)
        # #
        # prior p(z_2 | z_1)
        z1_flattened = z1.contiguous().view(-1, self.latent_dim_1 * 2 * 2)
        z2_mu_prior = self.prior_z_2_mu(z1_flattened).view(-1, self.latent_dim_2, 2, 2)
        
        mu, _ = self.decode(z2_mu_prior)  # Use only mu from your Decoder's output
        return mu
    
    @torch.no_grad()
    def sample_interpolate(self, x,x1, interpolate_pt = 10, device='cpu'):
        assert x.shape ==x1.shape
        _, _, z2_mu, z2_log_var, _, _, _ = self.encode(x)
        _, _, z2_mu_2, z2_log_var_2, _, _, _  = self.encode(x1)
        
        z1 = self.reparameterize(z2_mu, z2_log_var)
        z2 = self.reparameterize(z2_mu_2, z2_log_var_2)
        z_interpolate = self.interpolate(z1[:,None], z2[:,None])  # Linear interpolation in extra dim=1

        # Decode the interpo
    
        return self.sample_without_noise(z_interpolate.view(-1,*self.z_dim), size=interpolate_pt * interpolate_pt, device=device)
    @staticmethod
    def interpolate(z_start, z_end, size=10):
        # interpolate in extra dimension and then reshape
        weights = torch.linspace(0, 1, steps=size).view(1, size, 1, 1, 1).to(z_start.device)
        z_interp = (1 - weights) * z_start + weights * z_end
        return z_interp
    
        
        
