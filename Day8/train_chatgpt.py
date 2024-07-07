from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F    
import numpy as np

@dataclass
class GptConfig:
    block_size:int = 1024 # max length of sequence
    vocab_size:int = 50257
    n_layer:int = 12
    n_head:int = 12
    n_embed:int = 768


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate = "tanh")
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)
        

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0 
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))).view(1,1, config.block_size, config.block_size)  
        
        # B, nh, T, ns
    def forward(self,x):
        B, T, C = x.size()
        qkv  = self.c_attn(x)
        q, k,v = qkv.split(self.n_embed, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        attn = (k @ q.transpose(-2,-1)) * 1.0 (np.sqrt(k.size(-1)))
        attn = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(attn, dim = -1)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.y_proj(y)
        return y 


class Block:
    def __init__(self,config):
        super().__init__()
        self.ln1 = nn.LayerNorm(nn.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln1 = nn.LayerNorm(nn.n_embed)
        self.mlp = MLP(config)
        
        
    def forward(x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x 
    

class GPT2:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embeeding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_head),
        ))
        
        # there has to be a layer that has to predict one of the word from vocab
        # and this will be done using a linear layer that outputs the probability of the
        # vocab embeddings. 
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        

config = GptConfig()
ca = CasualSelfAttention(config)
x = torch.Tensor(np.arange(0,384,1))
ca(x.view(1,1,384))