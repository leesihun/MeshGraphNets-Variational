import torch.nn.init as init

import torch.nn as nn
import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from model.blocks import EdgeBlock, NodeBlock, HybridNodeBlock
from model.checkpointing import process_with_checkpointing

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Kaiming/He initialization for ReLU activation
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)


class MeshGraphNets(nn.Module):
    def __init__(self, config, device: str):
        super(MeshGraphNets, self).__init__()
        self.config = config
        self.device = device

        self.model = EncoderProcessorDecoder(config).to(device)
        self.model.apply(init_weights)
        print('MeshGraphNets model created successfully')

    def set_checkpointing(self, enabled: bool):
        """Enable or disable gradient checkpointing."""
        self.model.set_checkpointing(enabled)

    def forward(self, graph, debug=False):
        """
        Forward pass of the simulator.

        Expects pre-normalized inputs from the dataloader:
            - graph.x: normalized node features [N, input_var]
            - graph.edge_attr: normalized edge features [E, edge_var]
            - graph.y: normalized target delta (y_t+1 - x_t) [N, output_var]

        Returns:
            predicted: predicted normalized delta [N, output_var]
            target: normalized target delta [N, output_var] (None during rollout)
            kl: KL divergence loss (0.0 when use_vae=False or eval mode)
        """
        # Optional noise injection during training
        if self.training:
            noise_std = self.config.get('std_noise', 0.0)
            if noise_std > 0:
                noise = torch.randn_like(graph.x) * noise_std
                graph.x = graph.x + noise
        # Forward through encoder-processor-decoder
        predicted, kl = self.model(graph, debug=debug)

        return predicted, getattr(graph, 'y', None), kl

class EncoderProcessorDecoder(nn.Module):
    def __init__(self, config):
        super(EncoderProcessorDecoder, self).__init__()
        self.config = config

        self.message_passing_num = config['message_passing_num']
        self.edge_input_size = config['edge_var']
        self.latent_dim = config['latent_dim']
        self.use_checkpointing = config.get('use_checkpointing', False)
        self.use_world_edges = config.get('use_world_edges', False)

        # Compute actual node input size (physical features + optional node types)
        base_input_size = config['input_var']
        use_node_types = config.get('use_node_types', False)
        num_node_types = config.get('num_node_types', 0)
        if use_node_types and num_node_types > 0:
            self.node_input_size = base_input_size + num_node_types
            print(f"  Model input: {base_input_size} physical + {num_node_types} node types = {self.node_input_size}")
        else:
            self.node_input_size = base_input_size

        # Output is always physical features only (node types don't change)
        self.node_output_size = config['output_var']

        self.encoder = Encoder(
            self.edge_input_size, self.node_input_size, self.latent_dim,
            use_world_edges=self.use_world_edges
        )

        processer_list = []
        for _ in range(self.message_passing_num):
            processer_list.append(GnBlock(config, self.latent_dim, use_world_edges=self.use_world_edges))
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder = Decoder(self.latent_dim, self.node_output_size)

        # Variational encoder (optional)
        self.use_vae = config.get('use_vae', False)
        self.vae_latent_dim = config.get('vae_latent_dim', 32)
        if self.use_vae:
            self.vae_encoder = VariationalEncoder(
                self.node_output_size, self.latent_dim, self.vae_latent_dim)
            self.condition_proj = nn.Linear(
                self.latent_dim + self.vae_latent_dim, self.latent_dim)
            print(f"  Variational encoder: ENABLED (vae_latent_dim={self.vae_latent_dim})")

    def forward(self, graph, debug=False):
        # Save y and batch BEFORE encoder, which creates a new Data dropping them.
        # Use getattr: during rollout, graph has no .y (Data built manually).
        original_y = getattr(graph, 'y', None)
        original_batch = getattr(graph, 'batch', None)

        graph = self.encoder(graph)                                   # graph.y gone after this

        if debug:
            print(f"  After Encoder: x std={graph.x.std().item():.4f}, mean={graph.x.mean().item():.4f}")

        # VAE conditioning: encode ground-truth delta (train) or sample from prior (eval)
        kl = torch.tensor(0.0, device=graph.x.device)
        if self.use_vae:
            if self.training:
                z, mu, log_sq = self.vae_encoder(original_y, original_batch)
                kl = VariationalEncoder.kl_loss(mu, log_sq)
                batch_idx = original_batch
            else:
                # Inference: sample z from prior N(0, I)
                if original_batch is not None:
                    B, batch_idx = original_batch.max().item() + 1, original_batch
                else:                                                  # rollout: no .batch attr
                    B, batch_idx = 1, torch.zeros(
                        graph.x.shape[0], dtype=torch.long, device=graph.x.device)
                z = torch.randn(B, self.vae_latent_dim,
                                device=graph.x.device, dtype=graph.x.dtype)

            z_bcast = z[batch_idx]                                    # [N, vae_latent_dim]
            graph.x = self.condition_proj(torch.cat([graph.x, z_bcast], dim=-1))

            if debug:
                print(f"  After VAE conditioning: x std={graph.x.std().item():.4f}")

        if self.use_checkpointing and self.training:
            graph = process_with_checkpointing(self.processer_list, graph)
        else:
            for i, model in enumerate(self.processer_list):
                graph = model(graph)
                if debug and i == len(self.processer_list) - 1:
                    print(f"  After MP block {i}: x std={graph.x.std().item():.4f}, mean={graph.x.mean().item():.4f}")

        output = self.decoder(graph)
        if debug:
            print(f"  After Decoder: out std={output.std().item():.4f}, mean={output.mean().item():.4f}")
        return output, kl

    def set_checkpointing(self, enabled: bool):
        """Enable or disable gradient checkpointing."""
        self.use_checkpointing = enabled

def build_mlp(in_size, hidden_size, out_size, layer_norm=True, activation='relu', decoder=False):
    """
    Build a multi-layer perceptron following the original DeepMind MeshGraphNets architecture.

    Original paper (ICLR 2021): "all MLPs have two hidden layers with ReLU activation
    and the output layer (except for that of the decoding MLP) is normalized by LayerNorm"

    Structure: Input -> Hidden1 -> Hidden2 -> Output (3 Linear layers, 2 hidden)
    - ReLU activation between layers (original paper default)
    - LayerNorm on output (except decoder)
    """
    if activation == 'relu':
        activation_fn = nn.ReLU
    elif activation == 'gelu':
        activation_fn = nn.GELU
    elif activation == 'silu':
        activation_fn = nn.SiLU
    elif activation == 'tanh':
        activation_fn = nn.Tanh
    elif activation == 'sigmoid':
        activation_fn = nn.Sigmoid
    else:
        raise ValueError(f'Invalid activation function: {activation}')

    if layer_norm:
        # Standard MLP with 2 hidden layers + LayerNorm on output
        module = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, out_size),
            nn.LayerNorm(normalized_shape=out_size),
        )
    elif decoder:
        # Decoder: 2 hidden layers, NO LayerNorm on final output
        # This allows the decoder to output values with full range
        module = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, out_size)
        )

    return module

class VariationalEncoder(nn.Module):
    """
    Encodes ground-truth delta (graph.y) into a global stochastic latent z.
    Training: z = mu + sigma * eps (reparameterization trick).
    Inference: z ~ N(0, I) (sampled from prior, no ground-truth needed).
    """
    def __init__(self, input_dim, latent_dim, vae_latent_dim):
        super().__init__()
        self.node_encoder = build_mlp(input_dim, latent_dim, latent_dim)
        self.mu_head = nn.Linear(latent_dim, vae_latent_dim)
        self.log_sigma_sq_head = nn.Linear(latent_dim, vae_latent_dim)

    def forward(self, delta, batch):
        """
        Args:
            delta: [N_total, output_var] normalized target deltas (graph.y)
            batch: [N_total] PyG batch index mapping nodes to graphs
        Returns:
            z:            [B, vae_latent_dim] sampled latent
            mu:           [B, vae_latent_dim]
            log_sigma_sq: [B, vae_latent_dim]
        """
        h = self.node_encoder(delta)              # [N, latent_dim]
        h = global_mean_pool(h, batch)             # [B, latent_dim]
        mu = self.mu_head(h)                       # [B, vae_latent_dim]
        log_sq = self.log_sigma_sq_head(h)         # [B, vae_latent_dim]
        z = mu + torch.exp(0.5 * log_sq) * torch.randn_like(mu)
        return z, mu, log_sq

    @staticmethod
    def kl_loss(mu, log_sq):
        """KL(q(z|delta) || N(0,1)) = -0.5 * mean(1 + log_sigma_sq - mu^2 - exp(log_sigma_sq))"""
        return -0.5 * torch.mean(1.0 + log_sq - mu.pow(2) - log_sq.exp())


class Encoder(nn.Module):

    def __init__(self,
                edge_input_size,
                node_input_size,
                latent_dim,
                use_world_edges=False):
        super(Encoder, self).__init__()

        self.use_world_edges = use_world_edges
        self.eb_encoder = build_mlp(edge_input_size, latent_dim, latent_dim)
        self.nb_encoder = build_mlp(node_input_size, latent_dim, latent_dim)
        if use_world_edges:
            self.world_eb_encoder = build_mlp(edge_input_size, latent_dim, latent_dim)

    def forward(self, graph):
        node_attr, edge_attr = graph.x, graph.edge_attr
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        out = Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)

        if self.use_world_edges and hasattr(graph, 'world_edge_attr') and graph.world_edge_index.shape[1] > 0:
            world_edge_ = self.world_eb_encoder(graph.world_edge_attr)
            out.world_edge_attr = world_edge_
            out.world_edge_index = graph.world_edge_index
        elif self.use_world_edges:
            out.world_edge_attr = torch.zeros(0, edge_.shape[1], device=edge_.device)
            out.world_edge_index = torch.zeros(2, 0, dtype=torch.long, device=edge_.device)

        return out

class GnBlock(nn.Module):

    def __init__(self, config, latent_dim, use_world_edges=False):
        super(GnBlock, self).__init__()

        self.use_world_edges = use_world_edges
        self.residual_scale = config.get('residual_scale', 1.0)
        # Note: NVIDIA implementation uses full residual (scale=1.0) only for nodes, not edges

        eb_input_dim = 3 * latent_dim  # Sender, Receiver, edge latent dim
        eb_custom_func = build_mlp(eb_input_dim, latent_dim, latent_dim)
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)

        if use_world_edges:
            # World edge block (separate from mesh edge block)
            world_eb_custom_func = build_mlp(eb_input_dim, latent_dim, latent_dim)
            self.world_eb_module = EdgeBlock(custom_func=world_eb_custom_func)
            # Hybrid node block aggregates from both edge types
            nb_input_dim = 3 * latent_dim  # Node + mesh_agg + world_agg
            nb_custom_func = build_mlp(nb_input_dim, latent_dim, latent_dim)
            self.nb_module = HybridNodeBlock(custom_func=nb_custom_func)
        else:
            nb_input_dim = 2 * latent_dim  # Node + aggregated edges
            nb_custom_func = build_mlp(nb_input_dim, latent_dim, latent_dim)
            self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
        x_input = graph.x  # Save input for residual

        # Save world edge info before mesh edge block (which creates new Data)
        world_edge_index = graph.world_edge_index if self.use_world_edges and hasattr(graph, 'world_edge_index') else None
        world_edge_attr = graph.world_edge_attr if self.use_world_edges and hasattr(graph, 'world_edge_attr') and graph.world_edge_attr is not None and graph.world_edge_attr.shape[0] > 0 else None

        # Update mesh edge features (NO residual for edges - matches NVIDIA implementation)
        mesh_graph = self.eb_module(graph)
        edge_attr = mesh_graph.edge_attr  # Direct assignment, no residual

        # Update world edge features if present (NO residual)
        if self.use_world_edges and world_edge_attr is not None and world_edge_attr.shape[0] > 0:
            world_graph = Data(
                x=x_input,
                edge_attr=world_edge_attr,
                edge_index=world_edge_index
            )
            world_graph = self.world_eb_module(world_graph)
            updated_world_edge_attr = world_graph.edge_attr  # Direct assignment, no residual
        else:
            updated_world_edge_attr = world_edge_attr

        # Prepare graph for node block with updated edge features
        node_graph = Data(
            x=x_input,
            edge_attr=edge_attr,
            edge_index=mesh_graph.edge_index
        )
        if self.use_world_edges:
            node_graph.world_edge_attr = updated_world_edge_attr if updated_world_edge_attr is not None else torch.zeros(0, edge_attr.shape[1], device=x_input.device)
            node_graph.world_edge_index = world_edge_index if world_edge_index is not None else torch.zeros(2, 0, dtype=torch.long, device=x_input.device)

        # Update node features (aggregates from both edge types)
        node_graph = self.nb_module(node_graph)

        # Residual connection ONLY for nodes (matches NVIDIA implementation)
        x = x_input + self.residual_scale * node_graph.x

        out = Data(x=x, edge_attr=edge_attr, edge_index=node_graph.edge_index)
        if self.use_world_edges:
            out.world_edge_attr = updated_world_edge_attr if updated_world_edge_attr is not None else torch.zeros(0, edge_attr.shape[1], device=x.device)
            out.world_edge_index = world_edge_index if world_edge_index is not None else torch.zeros(2, 0, dtype=torch.long, device=x.device)

        return out

class Decoder(nn.Module):

    def __init__(self, latent_dim, node_output_size):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(latent_dim, latent_dim, node_output_size, layer_norm=False, decoder=True)

    def forward(self, graph):
        return self.decode_module(graph.x)
