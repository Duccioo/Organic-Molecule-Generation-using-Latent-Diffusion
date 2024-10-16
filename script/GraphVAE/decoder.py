import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class VAE_plain_DECODER(nn.Module):

    def __init__(self, embedding_size, h_size, y_size, e_size):
        super(VAE_plain_DECODER, self).__init__()
        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, embedding_size)
        self.decode_3 = nn.Linear(embedding_size, embedding_size)

        self.decode_adj_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_adj_2 = nn.Linear(embedding_size, h_size)
        # self.decode_2 = nn.Linear(embedding_size, y_size)

        self.relu = nn.ReLU()

        self.decode_node_features_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_node_features_2 = nn.Linear(embedding_size, y_size)

        self.decode_edges_features_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_edges_features_2 = nn.Linear(embedding_size, e_size)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, z):
        # decoder
        y = self.decode_1(z)
        y = self.relu(y)
        y = self.decode_2(y)
        # aggingo da v9.34 un altro layer per veder come va
        y = self.relu(y)
        y = self.decode_3(y)
        y = self.relu(y)

        semi_adj = self.decode_adj_1(y)
        semi_adj = self.relu(semi_adj)
        semi_adj = self.decode_adj_2(semi_adj)

        # decoder for node features
        n_features = self.decode_node_features_1(y)
        n_features = self.relu(n_features)
        n_features = self.decode_node_features_2(n_features)

        # decoder for edges features
        e_features = self.decode_edges_features_1(y)
        e_features = self.relu(e_features)
        e_features = self.decode_edges_features_2(e_features)

        return semi_adj, n_features, e_features


class VAE_plain_DECODER_2(nn.Module):
    def __init__(
        self,
        latent_embedding_size: int,
        h_size: int,
        n_size: int,
        e_size: int,
        num_layers: int = 3,
        num_layers_edge: int = 1,
        num_layers_node: int = 1,
        num_layers_adj: int = 1,
        dropout: float = 0.0001,
        growth_factor: float = 1.0,  # Nuovo parametro
    ):
        super(VAE_plain_DECODER_2, self).__init__()

        self.latent_embedding_size = latent_embedding_size
        self.h_size = h_size
        self.growth_factor = growth_factor

        # Decoder layers
        self.decoder_layers_init = nn.ModuleList()
        self.norm_layers_init = nn.ModuleList()

        self.decoder_layers_adj = nn.ModuleList()
        self.norm_layers_adj = nn.ModuleList()
        self.decoder_layers_node = nn.ModuleList()
        self.norm_layers_node = nn.ModuleList()
        self.decoder_layers_edge = nn.ModuleList()
        self.norm_layers_edge = nn.ModuleList()

        # Initial layers
        for i in range(num_layers):
            input_dim = latent_embedding_size if i == 0 else int(latent_embedding_size * (growth_factor**i))
            output_dim = int(latent_embedding_size * (growth_factor ** (i + 1)))
            self.decoder_layers_init.append(nn.Linear(input_dim, output_dim))
            self.norm_layers_init.append(nn.LayerNorm(output_dim))

        # Adjacency layers
        for i in range(num_layers_adj):
            input_dim = output_dim if i == 0 else int(h_size)
            output_dim_adj = int(h_size)
            self.decoder_layers_adj.append(nn.Linear(input_dim, output_dim_adj))
            self.norm_layers_adj.append(nn.LayerNorm(output_dim_adj))

        # Node layers
        for i in range(num_layers_node):
            input_dim = output_dim if i == 0 else int(n_size)
            output_dim_node = int(n_size)
            self.decoder_layers_node.append(nn.Linear(input_dim, output_dim_node))
            self.norm_layers_node.append(nn.LayerNorm(output_dim_node))

        # Edge layers
        for i in range(num_layers_edge):
            input_dim = output_dim if i == 0 else int(e_size)
            output_dim_edge = int(e_size)
            self.decoder_layers_edge.append(nn.Linear(input_dim, output_dim_edge))
            self.norm_layers_edge.append(nn.LayerNorm(output_dim_edge))

        # Output layers
        self.output_layer_adj = nn.Linear(int(h_size), h_size)
        self.output_layer_node = nn.Linear(int(n_size), n_size)
        self.output_layer_edge = nn.Linear(int(e_size), e_size)

        # Dropout
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

    def forward(self, z):
        x = z

        # Decoder layers with residual connections
        for decoder, norm in zip(self.decoder_layers_init, self.norm_layers_init):
            residual = x
            x = decoder(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if x.size() == residual.size():
                x += residual

        semi_adj = x
        for decoder, norm in zip(self.decoder_layers_adj, self.norm_layers_adj):
            residual = semi_adj
            semi_adj = decoder(semi_adj)
            semi_adj = norm(semi_adj)
            semi_adj = F.relu(semi_adj)
            semi_adj = F.dropout(semi_adj, p=self.dropout, training=self.training)
            # if semi_adj.size() == residual.size():
            #     semi_adj += residual

        edge = x
        for decoder, norm in zip(self.decoder_layers_edge, self.norm_layers_edge):
            residual = edge
            edge = decoder(edge)
            edge = norm(edge)
            edge = F.relu(edge)
            edge = F.dropout(edge, p=self.dropout, training=self.training)
            # if edge.size() == residual.size():
            #     edge += residual

        node = x
        for decoder, norm in zip(self.decoder_layers_node, self.norm_layers_node):
            residual = node
            node = decoder(node)
            node = norm(node)
            node = F.relu(node)
            node = F.dropout(node, p=self.dropout, training=self.training)
            # if node.size() == residual.size():
            #     node += residual

        # Output layer
        semi_adj = self.output_layer_adj(semi_adj)
        # semi_adj = F.sigmoid(semi_adj)
        edge = self.output_layer_edge(edge)
        # edge = F.sigmoid(edge)
        node = self.output_layer_node(node)
        # node = F.sigmoid(node)

        return semi_adj, node, edge
