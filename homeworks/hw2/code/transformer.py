import math
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # feature projections
        self.kqv_projection = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        

    def forward(self, x, k_cache=None, v_cache=None):
        B, T, C = x.size()  

        # calculate q, k v
        q, k, v = self.kqv_projection(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if k_cache is not None:
            assert v_cache is not None
            assert T == 1
            # concat previous cache with new k, v
            v = torch.cat([v_cache, v], dim=2) # (B, nh, 1 + T', hs)
            k = torch.cat([k_cache, k], dim=2) # (B, nh, 1 + T', hs)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, 1, 1 + T')
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        return self.proj(y), k, v


class Block(nn.Module):
    """Transfromer Block"""
    
    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        
        self.mlp_sequence = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x, k_cache=None, v_cache=None):
        _x = x
        x, k, v = self.attn(self.ln(x), k_cache, v_cache)
        x = _x + x
        x = x + self.mlp_sequence(x)
        return x, k, v


class Transformer(nn.Module):
    """Simple Transformer"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.token_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embd)
        self.transformer_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.layer_norm = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, start_idx=0, k_cache=None, v_cache=None):
        device = idx.device
        b, t = idx.size()
        assert (t <= self.block_size)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) + start_idx  

        tok_emb = self.token_embeddings(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.position_embeddings(pos)  # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb

        k_s = []
        v_s = []
        for idx, block in enumerate(self.transformer_blocks):
            if k_cache is not None:
                assert v_cache is not None
                k_i, v_i = k_cache[idx], v_cache[idx]
            else:
                k_i, v_i = None, None 
            x, k, v = block(x, k_i, v_i)
            k_s.append(k)
            v_s.append(v)

        logits = self.lm_head(self.layer_norm(x))
        return logits, k_s, v_s

    def loss(self, logits,k,v, x):
        targets = x[:, 1:] # make target a shifted version of the original
        logits = logits[:, :-1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        
        loss_dict = {
            "reconstruction_loss": torch.zeros(1,),
            "KLD": torch.zeros(1,), #We named vq_loss as KLD for trainer
            "loss": loss
        }
        return loss_dict

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens=None, temperature=1.0, cache=False, logit_mask=None
    ):
        """generate

        Args:
            idx (torch.Tensor): input indices
            max_new_tokens (int, optional): maximum number of tokens to generate. Defaults to None.
            temperature (float, optional): temperature for sampling. Defaults to 1.0.
            cache (bool, optional): whether to use caching. Defaults to False.
            logit_mask (torch.Tensor, optional): mask to apply to logits. Defaults to None, of shape (vocab_size,).
        """
        if max_new_tokens == None:
            max_new_tokens = self.block_size - idx.shape[-1]

        if logit_mask == None:
            logit_mask = torch.ones(self.vocab_size, dtype=int) 
            logit_mask[-1] = 0 # ignore last token, by default (usually sos)
            
        k_cache = None
        v_cache = None
           
        time_list = []
        for i in range(max_new_tokens):
            start_time = time.time()
            if cache:
                start_index = i
                idx_in = idx[:, -1:]
            else:
                start_index = 0
                idx_in = idx
            
            logits, k, v = self(idx_in, start_idx=start_index, k_cache=k_cache, v_cache=v_cache)
            if cache:
                k_cache = k
                v_cache = v

            # sample a token from the logits
            logits = logits[:, -1, :] / temperature
            if logit_mask is not None:
                logits[:, logit_mask == 0] = float("-inf") # mask out certain tokens

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            time_list.append(time.time() - start_time)

            # append sampled token
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, time_list