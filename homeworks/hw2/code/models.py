import torch
from torch import nn
from torch.nn import functional as F

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
    
    def sample_with_noise(self, size=100, device='cpu'):
        z = torch.randn(size, self.Decoder.latent_dim, device=device)
        mu, log_var = self.Decoder(z)  # Ensure your Decoder outputs both
        x_hat = self.reparameterize(mu, log_var)
        return x_hat

    def sample_without_noise(self, size=100, device='cpu'):
        z = torch.randn(size, self.Decoder.latent_dim, device=device)
        mu, _ = self.Decoder(z)  # Use only mu from your Decoder's output
        return mu
        
    