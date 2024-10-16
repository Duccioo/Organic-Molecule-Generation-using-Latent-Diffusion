import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F

from torch_geometric.nn import (
    TransformerConv,
    SAGEConv,
    LayerNorm,
    Set2Set,
    BatchNorm,
    global_mean_pool,
    global_max_pool,
)


class VAE_plain_ENCODER(nn.Module):

    def __init__(
        self, h_size, embedding_size, num_edges_features: int = 4, device="cpu"
    ):
        super(VAE_plain_ENCODER, self).__init__()
        self.device = device
        self.encode_11 = nn.Linear(h_size, embedding_size).to(device=device)  # mu
        self.encode_12 = nn.Linear(h_size, embedding_size).to(device=device)  # lsgms

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, h, edge_attr, edge_index, batch_index):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size())).to(self.device)
        z = eps * z_sgm + z_mu

        return z, z_mu, z_lsgms


class VAE_conv_ENCODER(nn.Module):

    def __init__(
        self,
        h_size: int,
        embedding_size: int,
        num_edges_features: int = 4,
        device="cpu",
    ):
        super(VAE_conv_ENCODER, self).__init__()
        self.device = device

        feature_size = h_size
        # print(h_size)
        self.encoder_embedding_size = embedding_size // 2
        self.edge_dim = num_edges_features
        self.latent_embedding_size = embedding_size

        # Encoder layers
        self.conv1 = TransformerConv(
            feature_size,
            self.encoder_embedding_size,
            heads=4,
            concat=False,
            beta=True,
            edge_dim=self.edge_dim,
        )
        self.bn1 = BatchNorm(self.encoder_embedding_size)
        self.conv2 = TransformerConv(
            self.encoder_embedding_size,
            self.encoder_embedding_size,
            heads=4,
            concat=False,
            beta=True,
            edge_dim=self.edge_dim,
        )
        self.bn2 = BatchNorm(self.encoder_embedding_size)
        self.conv3 = TransformerConv(
            self.encoder_embedding_size,
            self.encoder_embedding_size,
            heads=4,
            concat=False,
            beta=True,
            edge_dim=self.edge_dim,
        )
        self.bn3 = BatchNorm(self.encoder_embedding_size)
        self.conv4 = TransformerConv(
            self.encoder_embedding_size,
            self.encoder_embedding_size,
            heads=4,
            concat=False,
            beta=True,
            edge_dim=self.edge_dim,
        )

        # Pooling layers
        self.pooling = Set2Set(self.encoder_embedding_size, processing_steps=4)

        # Latent transform layers
        self.mu_transform = nn.Linear(
            self.encoder_embedding_size * 2, self.latent_embedding_size
        )
        self.logvar_transform = nn.Linear(
            self.encoder_embedding_size * 2, self.latent_embedding_size
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, x, edge_attr, edge_index, batch_index):
        # GNN layers
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_attr).relu()
        x = self.bn3(x)
        x = self.conv4(x, edge_index, edge_attr).relu()

        # Pool to global representation
        x = self.pooling(x, batch_index)

        # Latent transform layers
        z_mu = self.mu_transform(x)
        z_lsgms = self.logvar_transform(x)

        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size())).to(self.device)
        z = eps * z_sgm + z_mu

        return z, z_mu, z_lsgms


class VAE_conv_ENCODER_2(nn.Module):
    def __init__(
        self,
        h_size: int,
        embedding_size: int,
        num_edges_features: int = 4,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        pooling_type: str = "set2set",
        growth_factor: float = 2.0,  # incremento o decremento del numero dei neuroni
    ):
        super(VAE_conv_ENCODER_2, self).__init__()

        self.encoder_embedding_size = embedding_size // 2
        self.edge_dim = num_edges_features
        self.latent_embedding_size = embedding_size

        # Encoder layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(num_layers):
            input_dim = (
                h_size
                if i == 0
                else int(self.encoder_embedding_size * (growth_factor**i))
            )
            output_dim = int(self.encoder_embedding_size * (growth_factor ** (i + 1)))

            # metterci i lauer conv
            self.conv_layers.append(SAGEConv(input_dim, output_dim, normalize=True))

            # self.conv_layers.append(
            #     TransformerConv(
            #         input_dim,
            #         output_dim,
            #         heads=heads,
            #         concat=False,
            #         beta=True,
            #         edge_dim=self.edge_dim,
            #         dropout=dropout,
            #     )
            # )
            self.norm_layers.append(LayerNorm(output_dim))

        # Pooling layer
        last_output_dim = int(self.encoder_embedding_size * (growth_factor**num_layers))
        if pooling_type == "set2set":
            self.pooling = Set2Set(last_output_dim, processing_steps=4)
            pooling_output_dim = last_output_dim * 2
        elif pooling_type == "mean":
            self.pooling = global_mean_pool
            pooling_output_dim = last_output_dim
        elif pooling_type == "max":
            self.pooling = global_max_pool
            pooling_output_dim = last_output_dim
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

        # Latent transform layers
        self.mu_transform = nn.Linear(pooling_output_dim, self.latent_embedding_size)
        self.logvar_transform = nn.Linear(
            pooling_output_dim, self.latent_embedding_size
        )

        # Probability of an element getting zeroed.
        self.dropout = dropout

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_attr, edge_index, batch_index):
        # GNN layers with residual connections
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            # x = conv(x, edge_index, edge_attr)
            x = conv(x, edge_index)
            # x = norm(x)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool to global representation
        x = self.pooling(x, batch_index)

        # Latent transform layers
        z_mu = self.mu_transform(x)
        z_lsgms = self.logvar_transform(x)

        # Reparameterize
        noise_reduction_w = 1
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size())).to(x.device) * noise_reduction_w
        z = eps * z_sgm + z_mu

        return z, z_mu, z_lsgms


class VAE_conv_ENCODER_3(nn.Module):
    def __init__(
        self,
        h_size: int,
        embedding_size: int,
        num_edges_features: int = 4,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        pooling_type: str = "set2set",
    ):
        super(VAE_conv_ENCODER_3, self).__init__()

        self.encoder_embedding_size = embedding_size // 2
        self.edge_dim = num_edges_features
        self.latent_embedding_size = embedding_size

        self.num_layers = num_layers
        conv_layers = [
            TransformerConv(
                h_size,
                embedding_size // heads,
                heads=heads,
                beta=True,
                edge_dim=self.edge_dim,
            )
        ]
        conv_layers += [
            TransformerConv(
                embedding_size // heads,
                embedding_size // heads,
                heads=heads,
                beta=True,
                edge_dim=self.edge_dim,
            )
            for _ in range(num_layers - 2)
        ]
        # In the last layer, we will employ averaging for multi-head output by
        # setting concat to True.
        conv_layers.append(
            TransformerConv(
                embedding_size // heads,
                self.encoder_embedding_size,
                heads=heads,
                beta=True,
                concat=True,
                edge_dim=self.edge_dim,
            )
        )
        self.convs = torch.nn.ModuleList(conv_layers)

        # The list of layerNorm for each layer block.
        norm_layers = [
            torch.nn.LayerNorm(embedding_size) for _ in range(num_layers - 1)
        ]
        self.norms = torch.nn.ModuleList(norm_layers)

        # Probability of an element getting zeroed.
        self.dropout = dropout

        # Pooling layer
        if pooling_type == "set2set":
            self.pooling = Set2Set(self.encoder_embedding_size, processing_steps=4)
            pooling_output_dim = self.encoder_embedding_size * 2
        elif pooling_type == "mean":
            self.pooling = global_mean_pool
            pooling_output_dim = self.encoder_embedding_size
        elif pooling_type == "max":
            self.pooling = global_max_pool
            pooling_output_dim = self.encoder_embedding_size
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

        # Latent transform layers
        self.mu_transform = nn.Linear(pooling_output_dim, self.latent_embedding_size)
        self.logvar_transform = nn.Linear(
            pooling_output_dim, self.latent_embedding_size
        )

        # Probability of an element getting zeroed.
        self.dropout = dropout

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, x, edge_attr, edge_index, batch_index):
        # GNN layers with residual connections
        for i in range(self.num_layers - 1):
            # Construct the network as shown in the model architecture.
            # x = self.convs[i](x, edge_index, edge_attr)
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            # By setting training to self.training, we will only apply dropout
            # during model training.
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer, average multi-head output.
        x = self.convs[-1](x, edge_index)

        # Pool to global representation
        x = self.pooling(x, batch_index)

        # Latent transform layers
        z_mu = self.mu_transform(x)
        z_lsgms = self.logvar_transform(x)

        # Reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size())).to(x.device)
        z = eps * z_sgm + z_mu

        return z, z_mu, z_lsgms
