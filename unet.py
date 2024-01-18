from typing import List, Tuple
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, 
                 inplanes: int, 
                 planes: int, 
                 embedding_dim: int, 
                 dropout: float = 0.):
        super().__init__()
        self.gn1 = nn.GroupNorm(32, inplanes)
        self.silu = nn.SiLU(True)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.fc = nn.Linear(embedding_dim, planes)
        self.gn2 = nn.GroupNorm(32, planes, eps=1e-3)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)

        if inplanes != planes:
            self.conv3 = nn.Conv2d(inplanes, planes, 1)
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
    def __init__(self, inplanes: int):
        super().__init__()
        self.gn = nn.GroupNorm(32, inplanes)
        self.attn = nn.MultiheadAttention(inplanes, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.attn.in_proj_weight, 
            nonlinearity="linear"
        )
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, inputs: Tuple[Tensor, Tensor]):
        x, t_emb = inputs
        identity = x

        out = self.gn(x)
        out = out.flatten(2).permute(2, 0, 1)
        out = self.attn(out, out, out)[0]
        out = out.permute(1, 2, 0).reshape(*x.size())
        out += identity
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
    def __init__(self, inplanes: int):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, 3, 2, 1)
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
    def __init__(self, inplanes: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(inplanes, inplanes, 3, 1, 1)

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
    def __init__(
        self, 
        resolution: int,
        inplanes: int, 
        planes: List[int], 
        num_res_blocks: int, 
        dropout: float = 0.
    ) -> None:
        super().__init__()
        assert resolution in [32, 256]
        planes = [planes[0]] + planes 
        embedding_dim = 4 * planes[0]
        self.time_encoder = TimeEmbedding(embedding_dim)
        self.conv1 = nn.Conv2d(inplanes, planes[0], 3, 1, 1)
        self.conv2 = nn.Conv2d(planes[0], inplanes, 3, 1, 1)
        self.gn = nn.GroupNorm(32, planes[0])
        self.silu = nn.SiLU(True)

        encoder = []
        decoder = []
        _resolution = resolution
        for i in range(len(planes)-1):
            enc_block = nn.Sequential(
                ResBlock(
                    planes[i], 
                    planes[i+1],
                    embedding_dim,
                    dropout
                )
            )
            if _resolution == 16:
                enc_block.append(AttnBlock(planes[i+1]))

            enc_blocks = [enc_block]

            if i == 0:
                dec_block = nn.Sequential(
                    ResBlock(
                        planes[i] + planes[i+1], 
                        planes[i],
                        embedding_dim,
                        dropout
                    )
                )
            else:
                dec_block = nn.Sequential(
                        ResBlock(
                            planes[i] + planes[i+1],
                            planes[i+1],
                            embedding_dim,
                            dropout
                        )
                )
                if _resolution == 16:
                    dec_block.append(AttnBlock(planes[i+1]))

                dec_block.append(Upsample(planes[i+1]))

            dec_blocks = [dec_block]

            for _ in range(num_res_blocks-1):
                enc_block = nn.Sequential(
                    ResBlock(
                        planes[i+1], 
                        planes[i+1],
                        embedding_dim,
                        dropout
                    )
                )
                dec_block = nn.Sequential(
                    ResBlock(
                        2 * planes[i+1], 
                        planes[i+1],
                        embedding_dim,
                        dropout
                    )
                )
                if _resolution == 16:
                    enc_block.append(AttnBlock(planes[i+1]))
                    dec_block.append(AttnBlock(planes[i+1]))

                enc_blocks.append(enc_block)
                dec_blocks.append(dec_block)
                
            if i < len(planes) - 2:
                enc_blocks.append(Downsample(planes[i+1]))

                dec_block = nn.Sequential(
                    ResBlock(
                        planes[i+1] + planes[i+2],
                        planes[i+1],
                        embedding_dim,
                        dropout
                    )
                )
            else:
                dec_block = nn.Sequential(
                    ResBlock(
                        2 * planes[i+1],
                        planes[i+1],
                        embedding_dim,
                        dropout
                    )
                )

            if _resolution == 16:
                dec_block.append(AttnBlock(planes[i+1]))

            dec_blocks.append(dec_block)

            encoder.append(nn.ModuleList(enc_blocks))
            decoder.append(nn.ModuleList(reversed(dec_blocks)))

            _resolution = _resolution // 2

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(reversed(decoder))

        self.middle = nn.Sequential(
            ResBlock(
                planes[-1],
                planes[-1],
                embedding_dim,
                dropout
            ),
            AttnBlock(planes[-1]),
            ResBlock(
                planes[-1],
                planes[-1],
                embedding_dim,
                dropout
            )
        )

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
