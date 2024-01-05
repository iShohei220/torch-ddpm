from typing import List, Tuple
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 embedding_dim: int, 
                 dropout: float = 0.):
        super().__init__()
        self.gn1 = nn.GroupNorm(32, in_channels)
        self.silu = nn.SiLU(True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.fc = nn.Linear(embedding_dim, out_channels)
        self.gn2 = nn.GroupNorm(32, out_channels, eps=1e-3)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv3 = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        if self.conv3 is not None:
            nn.init.xavier_uniform_(self.conv3.weight)
            nn.init.zeros_(self.conv3.bias)

    def forward(self, inputs: Tuple[Tensor, Tensor]):
        x, t_emb = inputs
        identity = x

        out = self.gn1(x)
        out = self.silu(out)
        out = self.conv1(out)

        out += self.fc(t_emb).reshape(*out.size()[:2], 1, 1)

        out = self.gn2(out)
        out = self.silu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.conv3 is not None:
            identity = self.conv3(identity)

        out += identity
        out = (out, t_emb)

        return out


class AttnBlock(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gn = nn.GroupNorm(32, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, inputs: Tuple[Tensor, Tensor]):
        x, t_emb = inputs
        out = self.gn(x)
        out = out.flatten(2).permute(2, 0, 1)
        out = self.attn(out, out, out)[0]
        out = out.permute(1, 2, 0).reshape(*x.size())
        out = (out, t_emb)

        return out


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        assert embedding_dim % 8 == 0
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
                nn.Linear(embedding_dim // 4, embedding_dim),
                nn.SiLU(True),
                nn.Linear(embedding_dim, embedding_dim)
                )

    def forward(self, t: Tensor):
        out = math.log(10000) / (self.embedding_dim // 8 - 1)
        out = torch.exp(torch.arange(0, self.embedding_dim // 8, device=t.device) * - out)
        out = t.unsqueeze(-1) * out.unsqueeze(0)
        out = torch.cat([out.cos(), out.sin()], 1)
        out = self.net(out)

        return out


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, inputs: Tuple[Tensor, Tensor]):
        x, t_emb = inputs
        out = self.conv(x)
        out = (out, t_emb)

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, inputs: Tuple[Tensor, Tensor]):
        x, t_emb = inputs
        out = self.upsample(x)
        out = self.conv(out)
        out = (out, t_emb)

        return out


class UNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: List[int], 
                 num_res_blocks: int, dropout: float = 0.):
        super().__init__()
        hidden_channels = [hidden_channels[0]] + hidden_channels
        embedding_dim = 4 * hidden_channels[0]
        self.time_encoder = TimeEmbedding(embedding_dim)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], 3, 1, 1)
        self.conv2 = nn.Conv2d(hidden_channels[0], in_channels, 3, 1, 1)
        self.gn = nn.GroupNorm(32, hidden_channels[0])
        self.silu = nn.SiLU(True)

        encoder = []
        decoder = []
        for i in range(len(hidden_channels)-2):
            enc_blocks = [ResBlock(hidden_channels[i], 
                                   hidden_channels[i+1],
                                   embedding_dim)]
            if i == 0:
                dec_blocks = [ResBlock(hidden_channels[i]+hidden_channels[i+1], 
                                       hidden_channels[i],
                                       embedding_dim)]
            else:
                dec_blocks = [nn.Sequential(ResBlock(hidden_channels[i]+hidden_channels[i+1],
                                                     hidden_channels[i+1],
                                                     embedding_dim),
                                            Upsample(hidden_channels[i+1]))]

            for _ in range(num_res_blocks-1):
                enc_blocks.append(ResBlock(hidden_channels[i+1], 
                                           hidden_channels[i+1],
                                           embedding_dim))
                dec_blocks.append(ResBlock(2*hidden_channels[i+1], 
                                           hidden_channels[i+1],
                                           embedding_dim))
            enc_blocks.append(Downsample(hidden_channels[i+1]))
            dec_blocks.append(ResBlock(hidden_channels[i+1]+hidden_channels[i+2],
                                       hidden_channels[i+1],
                                       embedding_dim))
            encoder.append(nn.ModuleList(enc_blocks))
            decoder.append(nn.ModuleList(reversed(dec_blocks)))

        enc_blocks = [nn.Sequential(ResBlock(hidden_channels[-2], 
                                             hidden_channels[-1],
                                             embedding_dim),
                                    AttnBlock(hidden_channels[-1]))]
        dec_blocks = [nn.Sequential(ResBlock(hidden_channels[-2]+hidden_channels[-1], 
                                             hidden_channels[-1],
                                             embedding_dim),
                                    AttnBlock(hidden_channels[-1]),
                                    Upsample(hidden_channels[-1]))]

        for _ in range(num_res_blocks-1):
            enc_blocks.append(nn.Sequential(ResBlock(hidden_channels[-1], 
                                                     hidden_channels[-1],
                                                     embedding_dim),
                                            AttnBlock(hidden_channels[-1])))
            dec_blocks.append(nn.Sequential(ResBlock(2*hidden_channels[-1], 
                                                     hidden_channels[-1],
                                                     embedding_dim),
                                            AttnBlock(hidden_channels[-1])))

        dec_blocks.append(nn.Sequential(ResBlock(2*hidden_channels[-1], 
                                                 hidden_channels[-1],
                                                 embedding_dim),
                                        AttnBlock(hidden_channels[-1])))

        encoder.append(nn.ModuleList(enc_blocks))
        decoder.append(nn.ModuleList(reversed(dec_blocks)))

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(reversed(decoder))

        self.middle = nn.Sequential(ResBlock(hidden_channels[-1],
                                             hidden_channels[-1],
                                             embedding_dim),
                                    AttnBlock(hidden_channels[-1]),
                                    ResBlock(hidden_channels[-1],
                                             hidden_channels[-1],
                                             embedding_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor, t: Tensor):
        t_emb = self.time_encoder(t)

        h = self.conv1(x)
        hs = [h]

        for blocks in self.encoder:
            for block in blocks:
                h = block([h, t_emb])[0]
                hs.append(h)

        h = self.middle((h, t_emb))[0]

        for blocks in self.decoder:
            for block in blocks:
                h = block([torch.cat([h, hs.pop()], 1), t_emb])[0]

        out = self.gn(h)
        out = self.silu(out)
        out = self.conv2(out)

        return out
