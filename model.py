# my implement
import math
import inspect
import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from dataclasses import dataclass



@dataclass
class BertConfig:
    block_size: int = 256
    vocab_size: int = 161 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pad_token_id:int = 2


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def fill_pos_embedding(weight):
    n_pos = weight.shape[0]
    dim = weight.shape[1]

    weight.requires_grad = False
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    weight.data[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:,0::2]))
    weight.data[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:,1::2]))

    weight.detach_()


class CheckpointMixins:
    def load_state_dict(self, state_dict, **kwargs):
        pass
    
    def load_model(self, checkpoint_path, device):
        checkpoints = torch.load(checkpoint_path, map_location=device)        
        state_dict = checkpoints['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.load_state_dict(state_dict)
        return self



class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.char_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embd)
        fill_pos_embedding(self.pos_embeddings.weight)
        self.layer_norm = LayerNorm(config.n_embd, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.register_buffer(
            "position_ids", torch.arange(512).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids):
        B, seq_length = input_ids.shape
        char_embeddings = self.char_embeddings(input_ids)
        position_ids = self.position_ids[:, :seq_length]
        position_ids = position_ids.expand((input_ids.shape[0], -1))
        pos_embeddings = self.pos_embeddings(position_ids)
        embeddings = char_embeddings + pos_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=False)
        self.attention = nn.MultiheadAttention(config.n_embd, config.n_head, batch_first=True)
        self.drop_1 = nn.Dropout(config.dropout)
        self.ln_2 = LayerNorm(config.n_embd, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, attention_mask=None):
        x_norm = self.ln_1(x)
        sa_out, _ = self.attention(key=x_norm, value=x_norm, query=x_norm, key_padding_mask=attention_mask)
        sa_out = self.drop_1(sa_out)
        sa_out = sa_out + x
        return self.mlp(self.ln_2(sa_out)) + sa_out


class Bert(nn.Module, CheckpointMixins):
    def __init__(self, config):
        super().__init__()
        self.emb = EmbeddingLayer(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.emb.char_embeddings.weight = self.lm_head.weight # weight tying
        self.apply(self._init_weights)
        self.n_embd = config.n_embd
        self.pad_token_id = config.pad_token_id

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs, labels=None):
        x = self.emb(inputs)
        mask = inputs == self.pad_token_id
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)

        if labels is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(x) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss
    
    def get_embedding(self, inputs):
        B, T = inputs.shape
        x = self.emb(inputs)
        mask = inputs == self.pad_token_id
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return x
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.emb.pos_embeddings.weight.numel()
        return n_params
    

class Pooling(nn.Module):
    def forward(self, token_embeds, attention_mask): # B, T, D
        # reshape attention_mask to cover 768-dimension embeddings
        in_mask = attention_mask.unsqueeze(-1).expand(
            token_embeds.size()
        ).float()
        # perform mean-pooling but exclude padding tokens (specified by in_mask)
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
            in_mask.sum(1), min=1e-9
        )
        return pool


class SentenceEncoder(nn.Module, CheckpointMixins):
    def __init__(self, base_model, pad_token=-1):
        super(SentenceEncoder, self).__init__()
        self.base_model = base_model
        self.pad_token = pad_token
    
    def forward(self, input_ids):
        mask = input_ids != self.pad_token
        emb = self.base_model.get_embedding(input_ids)
        return self._mean_pooling(emb, mask)
    
    
    def _mean_pooling(self, token_embeds, attention_mask):
        # reshape attention_mask to cover 768-dimension embeddings
        in_mask = attention_mask.unsqueeze(-1).expand(
            token_embeds.size()
        ).float()
        # perform mean-pooling but exclude padding tokens (specified by in_mask)
        pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
            in_mask.sum(1), min=1e-9
        )
        return pool