import torch
from torch import nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Linear(2 * in_ch, out_ch)
            self.transform = nn.Linear(out_ch, out_ch)

        else:
            # down
            self.conv1 = nn.Linear(in_ch, out_ch)
            self.transform = nn.Linear(out_ch, out_ch)

        self.conv2 = nn.Linear(out_ch, out_ch)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):

        # First Conv
        h = self.relu(self.conv1(x))
        if h.shape[0] != 1:
            h = self.bnorm1(h)

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))

        # Add time channel
        h = h + time_emb
        # Second Conv

        h = self.relu(self.conv2(h))
        if h.shape[0] != 1:
            h = self.bnorm2(h)

        # Down or Upsample
        return self.transform(h)


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(
        self,
        latent_dimension: int = 5,
        down_channels: tuple = (4, 3, 2),
        time_emb_dim: int = 16,
    ):
        super().__init__()
        out_dim = latent_dimension

        up_channels = tuple(reversed(down_channels))

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # Initial projection
        self.conv0 = nn.Linear(latent_dimension, down_channels[0])

        # Downsample
        self.downs = nn.ModuleList(
            [
                Block(down_channels[i], down_channels[i + 1], time_emb_dim)
                for i in range(len(down_channels) - 1)
            ]
        )
        # Upsample
        self.ups = nn.ModuleList(
            [
                Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
                for i in range(len(up_channels) - 1)
            ]
        )

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Linear(up_channels[-1], out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)
