import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv2d, Linear, Flatten, ConvTranspose2d
from torch.nn import LayerNorm  # Ensure this is correctly imported
from models import *
import numpy as np
#Vector Quantizer



# VQVAE

class VectorQuantizer(nn.Module):
    def __init__(self, K, D, commitment_loss_weight):
        super(VectorQuantizer, self).__init__()
        self.K = K # Number of embeddings
        self.D = D #Embedding dimension
        self.commitment_loss_weight = commitment_loss_weight
        self.embedding = nn.Embedding(self.K, self.D)
        #initalize uniformly
        self.embedding.weight.data.uniform_(-1.0/K, 1.0/K)
        
    def forward(self, latents):
        
        #is B x C x H x W, permute to B x H x W x C
        latents = latents.permute(0, 2, 3, 1).contiguous()
        B, H, W, D = latents.shape
        flat_latents = latents.view(-1, self.D) #embed dim [B*H*W, D]
        
        # Distance between z and embeddings (N,D) v.s (K,D) -> (N,K)
        # latents**2 -> N,1, embed_w -> 1,K, 2*latents*embed_w -> N,K
        dist = torch.sum(flat_latents**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1).unsqueeze(0) - \
            2 * torch.matmul(flat_latents, self.embedding.weight.t())  # (N,K)
        
        
        # Get minimum distance index
        encoding_idxs = torch.argmin(dist, dim=1).unsqueeze(1) #(N,1)
        #create one-hot vecotr -> N,K
        encodings_one_hot = torch.zeros(B*H*W, self.K, device=latents.device)
        encodings_one_hot.scatter_(1, encoding_idxs, 1) #N,K
        
        # get quantized vector 
        quantized_latents = torch.matmul(encodings_one_hot, self.embedding.weight).view(B, H, W, D)
        
        #Loss
        embed_loss   = F.mse_loss(latents.detach(), quantized_latents, reduction='none')
        commit_loss = F.mse_loss(quantized_latents.detach(), latents, reduction='none')
        vq_loss =  embed_loss + self.commitment_loss_weight * commit_loss
        
        # Straight through gradient gradient from quantized flow trough latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        
        return vq_loss, quantized_latents.permute(0, 3, 1, 2).contiguous(), encoding_idxs.view(B,H,W) # [B, D, H, W])
        

class VQVAE(BaseVAE):
    def __init__(self, input_channels, K, D, beta=0.25, img_size = 32):
        super(VQVAE, self).__init__()
        
        #Define Vector Quantizer
        self.vector_quantizer = VectorQuantizer(K, D, beta)
        self.K = K
        self.D = D
        
        #define Encoder
        self.encoder = nn.Sequential(
            Conv2d(input_channels, 256, 4, stride=2, padding=1),# 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Conv2d(256, 256, 4, stride=2, padding=1), #8x8
            ResidualBlock(256),
            ResidualBlock(256),
        )
        
        #define Decoder
        self.decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ConvTranspose2d(256, 256, 4, stride=2, padding=1), #16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ConvTranspose2d(256, input_channels, 4, stride=2, padding=1), #32x32
            nn.Tanh()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        vq_loss, quantized_z, _ = self.vector_quantizer(z)
        x_recon = self.decode(quantized_z)
        return x_recon, vq_loss
    
    def loss(self, x_mean, vq_loss, x):
        # vq_loss = vq_loss.sum(dim=(1,2,3)).mean()
        
        vq_loss = vq_loss.sum(dim=(1,2,3)).mean() 
        reconstruction_loss =  F.mse_loss(x_mean, x, reduction='none').sum(dim=(1,2,3)).mean()
        total_loss = reconstruction_loss + vq_loss
        # print(vq_loss.item(),reconstruction_loss.item(), total_loss.item())
        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "KLD": vq_loss, #We named vq_loss as KLD for trainer
            "loss": total_loss
        }
        return loss_dict
    
    @torch.no_grad()
    def reconstruct(self, x):
        return self.forward(x)[0]
    
    
    @property
    def n_embeddings(self) -> int:
        """The size of the token vocabulary"""
        return self.K
    
    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize an image x.
        Args:
            x (np.ndarray, dtype=int): Image to quantize. shape=(batch_size, 28, 28, 3). Values in [0, 255].

        Returns:
            np.ndarray: Quantized image. shape=(batch_size, 8, 8). Values in [0, n_embeddings]
        """
        x_nomralized = normalize_images(x) #normalized to -1 1
        x = torch.tensor(x_nomralized, dtype = next(self.parameters()).dtype).permute(0, 3, 1, 2).to(next(self.parameters()).device) # B C H,W
        z_e = self.encode(x)
        _, _, z_index = self.vector_quantizer(z_e)
    
        return z_index.cpu().numpy()
        
    
    def decode_np(self, z_index: np.ndarray) -> np.ndarray:
        """Decode a quantized image.

        Args:
            z_index (np.ndarray, dtype=int): Quantized image. shape=(batch_size, 8, 8). Values in [0, n_embeddings].

        Returns:
            np.ndarray: Decoded image. shape=(batch_size, 32, 32, 3). Values in [0, 255].
        """
        z_index = torch.LongTensor(z_index).to(next(self.parameters()).device) 
        z_q = self.vector_quantizer.embedding(z_index) # shape B, H, W, D
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        x_recon_normalized = self.decode(z_q) #B, C, H, W
        x_recon = process_images(x_recon_normalized.cpu().detach().numpy())
        return x_recon #B,H,W,C
        
        
def process_images(images):
    images = images.transpose(0, 2, 3, 1)
    
    images = (images/2+0.5) * 255.0
    images = np.clip(images, 0, 255)
    
    # Convert to uint8
    images = images.astype(np.uint8)
    return images
def normalize_images(images):
    return (images / 255.0 - 0.5) * 2.0
        


if __name__ == "__main__":
    
    model = VQVAE(input_channels=3, K=128, D=256)
    x = torch.randn(10, 3, 32, 32)
    z = model(x)
   