import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np


from timm.models.vision_transformer import PatchEmbed, Attention
#################################################################################
# --------------Sine/Cosine Positional Embedding Functions-----------------------
#################################################################################
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# --------------------------EmBeddings----------------------------------
class TimeEmbedder(nn.Module):
    def __init__(self, hidden_dim, freq_embed_size=256):
        super().__init__()
        self.freq_embed_size = freq_embed_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )
        
        
        #  self.time_emb_mlp = nn.Sequential(
        #     nn.Linear(hidden_dims[0],  self.temb_channels),
        #     nn.SiLU(),
        #     nn.Linear(self.temb_channels,  self.temb_channels)
        # )
    @staticmethod
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
    def forward(self, t):
        t_embed = self.timestep_embedding(t, self.freq_embed_size)
        t_embed = self.mlp(t_embed)
        return t_embed

#learnable embeddings for class condition
class ClassEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_dim, dropout_prob = 0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.embed = nn.Embedding(num_classes+1, hidden_dim) #a null class for unconditional generation
        self.num_classes = num_classes
    def dropping_context(self, y):
        drop_ids = torch.rand(y.shape[0], device = y.device) <  self.dropout_prob
        updated_y = torch.where(drop_ids, self.num_classes, y)
        return updated_y
    def forward(self, y, training = False):
        
        #y is the label ehre
        if training:
            y = self.dropping_context(y)
        return self.embed(y)


# Reference https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------DiTBlock----------------------------------
class DiTBlock(nn.Module):
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # self.msa = nn.MultiheadAttention(hidden_size, num_heads, add_bias_kv=True, dropout=0.0)
        self.attn = Attention(hidden_size, num_heads , qkv_bias=True) #Note changing this attention matters
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias= True)
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
    def forward(self, x, c):
       # Given x (B x L x D), c (B x D)
        c = self.adaLN(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)

        h = self.norm1(x)
        h = modulate(h, shift_msa, scale_msa)
        # x = x + gate_msa.unsqueeze(1) * self.msa(h, h, h)[0]
        x = x + gate_msa.unsqueeze(1) * self.attn(h)
        h = self.norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)

        return x 

# DiTBlock(hidden_size, num_heads)
#     Given x (B x L x D), c (B x D)
#     c = SiLU()(c)
#     c = Linear(hidden_size, 6 * hidden_size)(c)
#     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)

#     h = LayerNorm(hidden_size, elementwise_affine=False)(x)
#     h = modulate(h, shift_msa, scale_msa)
#     x = x + gate_msa.unsqueeze(1) * Attention(hidden_size, num_heads)(h)

#     h = LayerNorm(hidden_size, elementwise_affine=False)(x)
#     h = modulate(h, shift_mlp, scale_mlp)
#     x = x + gate_mlp.unsqueeze(1) * MLP(hidden_size)(h)

#     return x
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps = 1e-6)
        self.fc = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm(x)
        x = modulate(x, shift, scale)
        x = self.fc(x)
        return x
# patch embedding
#https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L25

    
class DiT(nn.Module):
    def __init__(
        self, 
        input_shape, 
        patch_size, 
        hidden_size, 
        num_heads, 
        num_layers, 
        num_classes, 
        cfg_dropout_prob=0.1
    ):
        super().__init__()
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.cfg_dropout_prob = cfg_dropout_prob
        
        num_patch_size = input_shape[1] // patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, num_patch_size*num_patch_size, hidden_size), requires_grad=False) #shouldn't be changed!
        #patch embeding
       
        self.to_patch_embedding = PatchEmbed(
            8, patch_size, input_shape[0], hidden_size, bias=True
            )
        # nn.Sequential(
        #     nn.Conv2d(input_shape[0], hidden_size, kernel_size=patch_size, stride=patch_size),
        #     Rearrange('b c h w -> b (h w) c'),
        # )
        self.time_embedder = TimeEmbedder(hidden_size)
        self.class_embedder = ClassEmbedder(num_classes, hidden_size, cfg_dropout_prob)
        
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, patch_size, input_shape[0])

        self.weight_initialization()
    
    def weight_initialization(self):
        #transformer initializations
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.input_shape[1] // self.patch_size) #1, H*W, D
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        #projection
        w = self.to_patch_embedding.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.to_patch_embedding.proj.bias, 0)
        #TODO initialization according to paper (zero initialization?)
        #initialize embedding
        nn.init.normal_(self.class_embedder.embed.weight, std=0.02)
        #time embedding
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)
        
        
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.fc.weight, 0)
        nn.init.constant_(self.final_layer.fc.bias, 0)
    def patchify_flatten(self,x):
         # B x C x H x W -> B x (H // P * W // P) x D, P is patch_size
        return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
    def unpatchify(self, x):
        '''
         B x (H // P * W // P) x (P * P * C) -> B x C x H x W
        '''
        c, h, w = self.input_shape
        # Reconstruct the image from the patches
        x = x.view(x.shape[0], h//self.patch_size, w//self.patch_size, self.patch_size, self.patch_size, c)
        x = torch.einsum('b h w p q c -> b c h p w q', x)
        imgs = x.reshape(x.shape[0], c, h, w)
        # x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h//self.patch_size, w=w//self.patch_size, p1=self.patch_size, p2=self.patch_size, c=c)
        return imgs
            
    def forward(self, x, t, y):
        #   Given x (B x C x H x W) - image, y (B) - class label, t (B) - diffusion timestep
        x = self.to_patch_embedding(x)
        # x = self.patchify_flatten(x)
        x += self.pos_embed
        
        t = self.time_embedder(t)
        
        # if self.training:
        y = self.class_embedder(y, self.training)
        
        c = t + y
        
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x
    
    def forward_cfg(self, x, t, y, w):
        """Classifier Free guidances, w is the guidance vector
        """
        #expand x twice in batch
        # Prepare null class labels for unconditional generation
        y_null = torch.full_like(y, fill_value=self.class_embedder.num_classes)
        
        # Concatenate conditional and unconditional inputs
        x = torch.cat([x, x], dim=0)
        t = torch.cat([t, t], dim=0)
        y = torch.cat([y, y_null], dim=0)  # Use actual labels for first half, null labels for second
    
        
        eps = self.forward(x, t, y)
        cond_eps, uncond_eps = torch.split(eps, eps.shape[0]//2, dim=0)
        #use w to guide the conditional part
        eps_hat = uncond_eps + w * (cond_eps - uncond_eps)
        return eps_hat
        
# DiT(input_shape, patch_size, hidden_size, num_heads, num_layers, num_classes, cfg_dropout_prob)
#     Given x (B x C x H x W) - image, y (B) - class label, t (B) - diffusion timestep
#     x = patchify_flatten(x) # B x C x H x W -> B x (H // P * W // P) x D, P is patch_size
#     x += pos_embed # see get_2d_sincos_pos_embed

#     t = compute_timestep_embedding(t) # Same as in UNet
#     if training:
#         y = dropout_classes(y, cfg_dropout_prob) # Randomly dropout to train unconditional image generation
#     y = Embedding(num_classes + 1, hidden_size)(y)
#     c = t + y

#     for _ in range(num_layers):
#         x = DiTBlock(hidden_size, num_heads)(x, c)

#     x = FinalLayer(hidden_size, patch_size, out_channels)(x)
#     x = unpatchify(x) # B x (H // P * W // P) x (P * P * C) -> B x C x H x W
#     return x