import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv2d, Linear, Flatten, ConvTranspose2d

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
        log_var = torch.tanh(log_var)
        # Scale to [-10, 10]
        log_var = log_var * 10
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
        log_var = torch.tanh(log_var)
        # Scale to [-10, 10]
        log_var = log_var * 10
        return mean, log_var
        
class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, log_var = self.Encoder(x)
        z = self.reparameterize(mu, log_var)
        x_mean, x_log_var = self.Decoder(z)
        
        return  x_mean, x_log_var, mu, log_var
    
    
    @torch.no_grad()
    def sample_reconstruct(self, x, device='cpu'):
        mu, log_var = self.Encoder(x)
        z = self.reparameterize(mu, log_var)
       
        return self.sample_with_noise(z, device=device)
    
    @torch.no_grad()
    def sample_with_noise(self,z=None, size=100, device='cpu'):
        if z is None:
            z = torch.randn(size, self.Decoder.latent_dim, device=device)
        mu, log_var = self.Decoder(z)  # Ensure your Decoder outputs both
        x_hat = self.reparameterize(mu, log_var)
        return x_hat

    @torch.no_grad()
    def sample_without_noise(self, z=None, size=100, device='cpu'):
        if z is None:
            z = torch.randn(size, self.Decoder.latent_dim, device=device)
        mu, _ = self.Decoder(z)  # Use only mu from your Decoder's output
        return mu
    
    @torch.no_grad()
    def sample_interpolate(self, x,x1, interpolate_pt = 10, device='cpu'):
        
        assert x.shape ==x1.shape
        
        
        mu, _ = self.Encoder(x)
        mu_2, _ = self.Encoder(x1)
        
        z_interpolate = self.interpolate(mu[:,None], mu_2[:,None])  # Linear interpolation in extra dim=1

        # Decode the interpo
    
        return self.sample_with_noise(z_interpolate.view(-1,self.Decoder.latent_dim), size=interpolate_pt * interpolate_pt, device=device)
    
    @staticmethod
    def interpolate(z_start, z_end, size=10):
        weights = torch.linspace(0, 1, steps=size).unsqueeze(0).unsqueeze(-1).to(z_start.device)
        z_interp = (1 - weights) * z_start + weights * z_end
        return z_interp
    
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
            nn.Linear(4 * 4 * 256, 2 * latent_dim)
        )
    def forward(self, x):
        #
        output = self.net(x)
        # decopule to mu and log_var
        mu, log_var = output.chunk(2, dim=1)
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
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)  # Output: 32 x 32 (with 3 channels)
        )
    def forward(self, x):
        #
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)  # Reshape to (Batch, Channels, Height, Width)
        # decopule to mu and log_var
        x = self.conv_transpose(x)
        mu = torch.sigmoid(x)
        log_var = torch.zeros_like(mu)
        # Scale to [-10, 10]
        # log_var = torch.tanh(log_var)
        # # Scale to [-10, 10]
        # log_var = log_var * 10
        # log_var = log_var * 10
        return mu, log_var