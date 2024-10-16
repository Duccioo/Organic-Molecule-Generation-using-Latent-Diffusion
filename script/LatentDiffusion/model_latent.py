import torch
import torch.nn as nn


# ---
from LatentDiffusion.linear_noise_scheduled import LinearNoiseScheduler


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # device = time.device
        # half_dim = self.dim // 2
        # embeddings = math.log(10000) / (half_dim - 1)
        # embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # embeddings = time[:, None] * embeddings[None, :]
        # embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        assert self.dim % 2 == 0, "time embedding dimension must be divisible by 2"
        # factor = 10000^(2i/d_model)
        factor = 10000 ** (
            (
                torch.arange(start=0, end=self.dim // 2, dtype=torch.float32, device=time.device)
                / (self.dim // 2)
            )
        )

        # pos / factor

        # timesteps B -> B, 1 -> B, temb_dim
        embeddings = time[:, None].repeat(1, self.dim // 2) / factor
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class Block_linear(nn.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim, up=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Linear(time_emb_dim, out_dim)
        if up:
            self.conv1 = nn.Linear(2 * in_dim, out_dim)
            self.transform = nn.Linear(out_dim, out_dim)
        else:
            self.conv1 = nn.Linear(in_dim, out_dim)
            self.transform = nn.Linear(out_dim, out_dim)
        self.conv2 = nn.Linear(out_dim, out_dim)
        self.bnorm1 = nn.BatchNorm1d(out_dim)
        self.bnorm2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, t):

        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t.squeeze()))

        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class UNet(nn.Module):
    def __init__(self, in_channels, time_dim=256, depths=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.depths = depths

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.inc = nn.Linear(in_channels, depths[0])

        self.down_layers = nn.ModuleList(
            [Block_linear(depths[i], depths[i + 1], time_dim) for i in range(len(depths) - 1)]
        )

        self.up_layers = nn.ModuleList(
            [Block_linear(depths[i], depths[i - 1], time_dim, up=True) for i in range(len(depths) - 1, 0, -1)]
        )

        self.out = nn.Linear(depths[0], in_channels)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        # t = self.sinusoidal_embedding(timestep)
        # t = self.time_mlp(t)

        x = self.inc(x)

        residuals = []
        residuals.append(x)
        for down in self.down_layers:
            x = down(x, t)
            residuals.append(x)

        for up in self.up_layers:
            residual = residuals.pop()

            x = torch.cat((x, residual), dim=1)
            x = up(x, t)

        return self.out(x)


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        encoder_decoder,
        latent_dim,
        unet_in_channels,
        time_dim=256,
        num_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        unet_depths=[64, 128, 256, 512, 1024],
    ):
        super(LatentDiffusionModel, self).__init__()

        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.scheduler = LinearNoiseScheduler(num_timesteps, beta_start, beta_end)
        self.encoder_decoder = encoder_decoder
        self.latent_dim = latent_dim

        self.unet = UNet(unet_in_channels, time_dim, unet_depths)
        # self.unet = GeneralUnet(unet_in_channels, time_dim=time_dim)

    def forward(self, z, t):
        # Encode input to latent space
        # z = self.encoder_decoder.encode(x)

        # Predict noise
        noise_pred = self.unet(z, t)

        return noise_pred

    def add_noise(self, z, t):
        # Encode input to latent space
        # z, _, _ = self.encoder_decoder.encode(x)

        # Generate noise
        noise = torch.randn_like(z)

        # Calculate alpha and beta
        alpha = 1 - t
        beta = t

        # Add noise to the latent representation
        noisy_z = torch.sqrt(alpha) * z + torch.sqrt(beta) * noise

        return noisy_z, noise

    def add_noise_2(self, z, t):
        # https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/scheduler/linear_noise_scheduler.py

        # Generate noise
        noise = torch.randn_like(z).to(z.device)
        noisy_z = self.scheduler.add_noise(z, noise, t)

        return noisy_z, noise

    def decode(self, noise_latent, steps=500, treshold_adj: float = 0.5, treshold_diag: float = 0.5):
        """
        Decodes the given noise latent to an image space.

        Parameters:
        -----------
        noise_latent: torch.Tensor
            The noise latent to be decoded.
        steps: int (default=100)
            The number of steps to be taken in the diffusion process.

        Returns:
        --------
        torch.Tensor
            The decoded image space.
        """
        with torch.no_grad():
            # Reverse diffusion process
            for i in reversed(range(steps)):
                # t = torch.ones(noise.shape[0], dtype=torch.float, device=noise.device) * i / steps

                noise_pred = self(noise_latent, torch.as_tensor(i).unsqueeze(0).to(noise_latent.device))
                # Use scheduler to get x0 and xt-1
                noise_latent, _ = self.scheduler.sample_prev_timestep(
                    noise_latent, noise_pred, torch.as_tensor(i).to(noise_latent.device)
                )
            data = self.encoder_decoder.generate(noise_latent, treshold_adj, treshold_diag)

        # Decode latent to image space
        return data

    def recostruct(self, data, steps: int = 100, treshold_adj: float = 0.5, treshold_diag: float = 0.5):

        with torch.no_grad():
            z, _, _ = self.encoder_decoder.encoder(data)

            data = self.decode(z, steps=steps, treshold_adj=treshold_adj, treshold_diag=treshold_diag)

        return data
