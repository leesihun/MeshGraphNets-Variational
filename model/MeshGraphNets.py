import torch.nn.init as init

import torch.nn as nn
import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from general_modules.edge_features import EDGE_FEATURE_DIM
from model.blocks import EdgeBlock, NodeBlock, HybridNodeBlock
from model.checkpointing import process_with_checkpointing
from model.coarsening import pool_features, unpool_features

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

        # Scale decoder's last layer for better initial predictions.
        # T>1 (delta prediction): scale to ~0 ("predict no change" prior)
        # T=1 (static prediction): keep Kaiming init (targets are full displacements, not small deltas)
        num_timesteps = config.get('num_timesteps', None)
        if num_timesteps is None or num_timesteps > 1:
            with torch.no_grad():
                last_layer = self.model.decoder.decode_module[-1]
                last_layer.weight.mul_(0.01)

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
            target: normalized target delta [N, output_var]
        """
        # Noise injection during training (node + edge noise for regularization)
        if self.training:
            noise_std = self.config.get('std_noise', 0.0)
            if noise_std > 0:
                output_var = self.config['output_var']
                # Node noise: physical features only (not node type one-hot)
                noise = torch.randn(graph.x.shape[0], output_var,
                                    device=graph.x.device, dtype=graph.x.dtype) * noise_std
                noise_padded = torch.zeros_like(graph.x)
                noise_padded[:, :output_var] = noise
                graph.x = graph.x + noise_padded
                # Target correction (DeepMind: delta -= gamma * noise)
                noise_gamma = self.config.get('noise_gamma', 0.1)
                noise_std_ratio = self.config.get('noise_std_ratio', None)
                if noise_std_ratio is not None:
                    ratio = torch.tensor(noise_std_ratio, device=graph.x.device,
                                         dtype=graph.x.dtype)
                    graph.y = graph.y - noise_gamma * noise * ratio
                # Edge noise: same std on all edge features (normalized space)
                edge_noise = torch.randn_like(graph.edge_attr) * noise_std
                graph.edge_attr = graph.edge_attr + edge_noise
        # Forward through encoder-processor-decoder
        predicted, kl = self.model(graph, debug=debug)

        return predicted, graph.y, kl

class EncoderProcessorDecoder(nn.Module):
    def __init__(self, config):
        super(EncoderProcessorDecoder, self).__init__()
        self.config = config

        self.message_passing_num = config['message_passing_num']
        self.edge_input_size = int(config['edge_var'])
        if self.edge_input_size != EDGE_FEATURE_DIM:
            raise ValueError(
                f"edge_var must be {EDGE_FEATURE_DIM}, got {self.edge_input_size}"
            )
        self.latent_dim = config['latent_dim']
        self.use_checkpointing = config.get('use_checkpointing', False)
        self.use_world_edges = config.get('use_world_edges', False)
        self.use_multiscale = config.get('use_multiscale', False)

        # Compute actual node input size (physical + positional + optional node types)
        base_input_size = config['input_var']
        num_pos_features = int(config.get('positional_features', 0))
        base_input_size += num_pos_features
        use_node_types = config.get('use_node_types', False)
        num_node_types = config.get('num_node_types', 0)
        if use_node_types and num_node_types > 0:
            self.node_input_size = base_input_size + num_node_types
            print(f"  Model input: {config['input_var']} physical + {num_pos_features} positional + {num_node_types} node types = {self.node_input_size}")
        else:
            self.node_input_size = base_input_size
            if num_pos_features > 0:
                print(f"  Model input: {config['input_var']} physical + {num_pos_features} positional = {self.node_input_size}")

        # Output is always physical features only (node types don't change)
        self.node_output_size = config['output_var']

        self.encoder = Encoder(
            self.edge_input_size, self.node_input_size, self.latent_dim,
            use_world_edges=self.use_world_edges
        )

        if not self.use_multiscale:
            # ── Original flat processor ──────────────────────────────────────
            processer_list = []
            for _ in range(self.message_passing_num):
                processer_list.append(GnBlock(config, self.latent_dim, use_world_edges=self.use_world_edges))
            self.processer_list = nn.ModuleList(processer_list)
        else:
            # ── Hierarchical N-level V-cycle processor ───────────────────────
            L = int(config.get('multiscale_levels', 1))
            self.multiscale_levels = L

            # Parse mp_per_level or construct from legacy keys
            mp_per_level = config.get('mp_per_level', None)
            if mp_per_level is None:
                fine_pre = int(config.get('fine_mp_pre', 5))
                coarse_mp = int(config.get('coarse_mp_num', 5))
                fine_post = int(config.get('fine_mp_post', 5))
                mp_per_level = [fine_pre] + [coarse_mp] + [fine_post]
            if not isinstance(mp_per_level, list):
                mp_per_level = [int(mp_per_level)]
            else:
                mp_per_level = [int(x) for x in mp_per_level]

            expected_len = 2 * L + 1
            if len(mp_per_level) != expected_len:
                raise ValueError(
                    f"mp_per_level must have {expected_len} entries for {L} levels, "
                    f"got {len(mp_per_level)}: {mp_per_level}"
                )

            # Print V-cycle structure
            parts = []
            for i in range(L):
                parts.append(f"pre[{i}]={mp_per_level[i]}")
            parts.append(f"coarsest={mp_per_level[L]}")
            for i in range(L - 1, -1, -1):
                parts.append(f"post[{i}]={mp_per_level[2 * L - i]}")
            print(f"  Multiscale V-cycle ({L} levels): {', '.join(parts)}")

            coarse_config = dict(config)
            coarse_config['use_world_edges'] = False

            # Build per-level pre-blocks, post-blocks, edge encoders, skip projections
            self.pre_blocks = nn.ModuleList()
            self.post_blocks = nn.ModuleList()
            self.coarse_eb_encoders = nn.ModuleList()
            self.skip_projs = nn.ModuleList()

            for i in range(L):
                pre_count = mp_per_level[i]
                post_count = mp_per_level[2 * L - i]
                use_we = self.use_world_edges if i == 0 else False
                cfg = config if i == 0 else coarse_config

                self.pre_blocks.append(nn.ModuleList([
                    GnBlock(cfg, self.latent_dim, use_world_edges=use_we)
                    for _ in range(pre_count)
                ]))
                self.post_blocks.append(nn.ModuleList([
                    GnBlock(cfg, self.latent_dim, use_world_edges=use_we)
                    for _ in range(post_count)
                ]))
                self.coarse_eb_encoders.append(
                    build_mlp(self.edge_input_size, self.latent_dim, self.latent_dim)
                )
                self.skip_projs.append(nn.Linear(2 * self.latent_dim, self.latent_dim))

            # Bipartite message passing unpool (learned coarse→fine)
            self.bipartite_unpool = config.get('bipartite_unpool', False)
            if self.bipartite_unpool:
                from model.blocks import UnpoolBlock
                self.unpool_blocks = nn.ModuleList([
                    UnpoolBlock(self.latent_dim, build_mlp)
                    for _ in range(L)
                ])

            # Coarsest level blocks
            coarsest_count = mp_per_level[L]
            self.coarsest_blocks = nn.ModuleList([
                GnBlock(coarse_config, self.latent_dim, use_world_edges=False)
                for _ in range(coarsest_count)
            ])

        self.decoder = Decoder(self.latent_dim, self.node_output_size)

        # Gated encoder→decoder skip connection: learns when to use encoder features
        self.skip_gate = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.Sigmoid()
        )

        # Variational conditioning (conditional VAE — encodes target delta during training)
        self.use_vae = config.get('use_vae', False)
        if self.use_vae:
            self.vae_latent_dim = int(config.get('vae_latent_dim', 32))
            self.vae_encoder = VariationalEncoder(
                self.node_output_size, self.latent_dim, self.vae_latent_dim
            )
            self.condition_proj = nn.Linear(self.latent_dim + self.vae_latent_dim, self.latent_dim)
            print(f"  VAE: ENABLED (z_dim={self.vae_latent_dim})")

    def _vae_condition(self, encoder_x, original_y, original_batch):
        """
        Apply VAE conditioning.
          Training + y available: encode y → (z, mu, logvar), reparameterize, return (x_cond, kl).
          Eval or y=None:         sample z ~ N(0, I), return (x_cond, 0.0).
        z is broadcast per-node then concatenated with encoder_x and projected back to latent_dim.
        """
        N = encoder_x.shape[0]
        device = encoder_x.device
        dtype = encoder_x.dtype

        if self.training and original_y is not None:
            batch = (original_batch if original_batch is not None
                     else torch.zeros(N, dtype=torch.long, device=device))
            z, mu, logvar = self.vae_encoder(original_y, batch)
            kl = VariationalEncoder.kl_loss(mu, logvar)
        else:
            B = int(original_batch.max().item()) + 1 if original_batch is not None else 1
            z = torch.randn(B, self.vae_latent_dim, device=device, dtype=dtype)
            kl = 0.0

        batch_bc = (original_batch if original_batch is not None
                    else torch.zeros(N, dtype=torch.long, device=device))
        z_per_node = z[batch_bc]  # [N, vae_latent_dim]
        x_conditioned = self.condition_proj(torch.cat([encoder_x, z_per_node], dim=-1))
        return x_conditioned, kl

    def forward(self, graph, debug=False):
        if not self.use_multiscale:
            # ── Original flat path ────────────────────────────────────────────
            # Save y and batch BEFORE encoder (encoder drops them)
            original_y = getattr(graph, 'y', None)
            original_batch = getattr(graph, 'batch', None)

            graph = self.encoder(graph)
            encoder_x = graph.x  # save pre-VAE encoder output for skip connection
            if debug:
                print(f"  After Encoder: x std={graph.x.std().item():.4f}, mean={graph.x.mean().item():.4f}")

            # VAE conditioning (after encoder, before processor)
            kl = 0.0
            if self.use_vae:
                x_conditioned, kl = self._vae_condition(graph.x, original_y, original_batch)
                graph.x = x_conditioned

            if self.use_checkpointing and self.training:
                graph = process_with_checkpointing(self.processer_list, graph)
            else:
                for i, model in enumerate(self.processer_list):
                    graph = model(graph)
                    if debug and i == len(self.processer_list) - 1:
                        print(f"  After MP block {i}: x std={graph.x.std().item():.4f}, mean={graph.x.mean().item():.4f}")
            # Gated skip: blend encoder features back into processed output
            gate = self.skip_gate(torch.cat([graph.x, encoder_x], dim=-1))
            graph = Data(x=graph.x + gate * encoder_x, edge_attr=graph.edge_attr, edge_index=graph.edge_index)
            output = self.decoder(graph)
            if debug:
                print(f"  After Decoder: out std={output.std().item():.4f}, mean={output.mean().item():.4f}")
            return output, kl

        # ── Hierarchical N-level V-cycle path ─────────────────────────────────
        L = self.multiscale_levels

        # Save y and batch BEFORE encoder (encoder drops them)
        original_y = getattr(graph, 'y', None)
        original_batch = getattr(graph, 'batch', None)

        # Extract per-level topology BEFORE encoder (encoder drops custom attrs)
        level_data = {}
        for i in range(L):
            ftc_key = f'fine_to_coarse_{i}'
            if not hasattr(graph, ftc_key):
                # Graph has fewer levels than model (degeneracy) — stop here
                break
            ld = {
                'ftc': graph[ftc_key],
                'c_ei': graph[f'coarse_edge_index_{i}'],
                'c_ea': graph[f'coarse_edge_attr_{i}'],
                'n_c': int(graph[f'num_coarse_{i}'].sum()),
            }
            if self.use_multiscale and getattr(self, 'bipartite_unpool', False):
                ld['up_ei'] = graph[f'unpool_edge_index_{i}']
                ld['coarse_centroid'] = graph[f'coarse_centroid_{i}']
                ld['fine_pos'] = graph.pos if i == 0 else graph[f'coarse_centroid_{i - 1}']
            level_data[i] = ld
        actual_levels = len(level_data)

        # Encode fine graph
        graph = self.encoder(graph)
        encoder_x = graph.x  # save pre-VAE encoder output for skip connection
        if debug:
            print(f"  [MS] After Encoder: x std={graph.x.std().item():.4f}")

        # VAE conditioning at fine level (after encoder, before V-cycle)
        kl = 0.0
        if self.use_vae:
            x_conditioned, kl = self._vae_condition(graph.x, original_y, original_batch)
            graph.x = x_conditioned

        # ── DESCENDING ARM (fine → coarse) ───────────────────────────────────
        skip_states = []
        current_graph = graph

        for i in range(actual_levels):
            # Run pre-blocks at level i
            if self.use_checkpointing and self.training:
                current_graph = process_with_checkpointing(self.pre_blocks[i], current_graph)
            else:
                for block in self.pre_blocks[i]:
                    current_graph = block(current_graph)

            if debug:
                print(f"  [MS] After pre[{i}] ({len(self.pre_blocks[i])} blocks): x std={current_graph.x.std().item():.4f}")

            # Save skip connection state
            skip_states.append({
                'x': current_graph.x,
                'edge_attr': current_graph.edge_attr,
                'edge_index': current_graph.edge_index,
                'w_attr': getattr(current_graph, 'world_edge_attr', None) if i == 0 and self.use_world_edges else None,
                'w_idx': getattr(current_graph, 'world_edge_index', None) if i == 0 and self.use_world_edges else None,
            })

            # Pool: level i → level i+1
            ld = level_data[i]
            h_coarse = pool_features(current_graph.x, ld['ftc'], ld['n_c'])
            e_coarse = self.coarse_eb_encoders[i](ld['c_ea'])
            current_graph = Data(x=h_coarse, edge_attr=e_coarse, edge_index=ld['c_ei'])

            if debug:
                print(f"  [MS] After pool[{i}]: {skip_states[-1]['x'].shape[0]} → {h_coarse.shape[0]} nodes")

        # ── COARSEST LEVEL ────────────────────────────────────────────────────
        if self.use_checkpointing and self.training:
            current_graph = process_with_checkpointing(self.coarsest_blocks, current_graph)
        else:
            for block in self.coarsest_blocks:
                current_graph = block(current_graph)

        if debug:
            print(f"  [MS] After coarsest ({len(self.coarsest_blocks)} blocks): x std={current_graph.x.std().item():.4f}")

        # ── ASCENDING ARM (coarse → fine) ─────────────────────────────────────
        for i in range(actual_levels - 1, -1, -1):
            # Unpool: level i+1 → level i
            ld = level_data[i]
            if getattr(self, 'bipartite_unpool', False):
                src, dst = ld['up_ei']
                rel_pos = ld['fine_pos'][dst] - ld['coarse_centroid'][src]
                h_up = self.unpool_blocks[i](
                    h_coarse=current_graph.x,
                    h_fine_skip=skip_states[i]['x'],
                    unpool_edge_index=ld['up_ei'],
                    rel_pos=rel_pos,
                )
            else:
                h_up = unpool_features(current_graph.x, ld['ftc'])

            # Skip connection merge
            skip = skip_states[i]
            h_merged = self.skip_projs[i](torch.cat([skip['x'], h_up], dim=-1))

            # Reconstruct graph at level i with saved edge topology
            current_graph = Data(x=h_merged, edge_attr=skip['edge_attr'], edge_index=skip['edge_index'])
            if i == 0 and self.use_world_edges and skip['w_attr'] is not None:
                current_graph.world_edge_attr = skip['w_attr']
                current_graph.world_edge_index = skip['w_idx']

            # Run post-blocks at level i
            if self.use_checkpointing and self.training:
                current_graph = process_with_checkpointing(self.post_blocks[i], current_graph)
            else:
                for block in self.post_blocks[i]:
                    current_graph = block(current_graph)

            if debug:
                print(f"  [MS] After post[{i}] ({len(self.post_blocks[i])} blocks): x std={current_graph.x.std().item():.4f}")

        # Gated skip: blend encoder features back into processed output
        gate = self.skip_gate(torch.cat([current_graph.x, encoder_x], dim=-1))
        current_graph = Data(x=current_graph.x + gate * encoder_x, edge_attr=current_graph.edge_attr, edge_index=current_graph.edge_index)
        # Decode
        output = self.decoder(current_graph)
        if debug:
            print(f"  [MS] After Decoder: out std={output.std().item():.4f}")
        return output, kl

    def set_checkpointing(self, enabled: bool):
        """Enable or disable gradient checkpointing."""
        self.use_checkpointing = enabled

def build_mlp(in_size, hidden_size, out_size, layer_norm=True, activation='silu', decoder=False):
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
    """Encodes ground-truth target delta into a global stochastic latent z (cVAE encoder)."""

    def __init__(self, node_output_size, latent_dim, vae_latent_dim):
        super().__init__()
        self.encoder_mlp = build_mlp(node_output_size, latent_dim, latent_dim)
        self.mu_head = nn.Linear(latent_dim, vae_latent_dim)
        self.logvar_head = nn.Linear(latent_dim, vae_latent_dim)

    def forward(self, y, batch):
        """
        Args:
            y:     [N, node_output_size] per-node target delta
            batch: [N] PyG batch assignment index
        Returns:
            z:      [B, vae_latent_dim] reparameterized latent
            mu:     [B, vae_latent_dim]
            logvar: [B, vae_latent_dim]
        """
        h = self.encoder_mlp(y)                # [N, latent_dim]
        h_pooled = global_mean_pool(h, batch)  # [B, latent_dim]
        mu = self.mu_head(h_pooled)            # [B, vae_latent_dim]
        logvar = self.logvar_head(h_pooled)    # [B, vae_latent_dim]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    @staticmethod
    def kl_loss(mu, logvar):
        """KL(q||p) = -1/2 * mean(1 + logvar - mu^2 - exp(logvar))"""
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


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
        self.use_pairnorm = config.get('use_pairnorm', False)
        # Residual connections applied to both nodes and edges (matches DeepMind original)

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

        # Update mesh edge features (raw MLP output, residual applied after node update)
        mesh_graph = self.eb_module(graph)
        edge_mlp_out = mesh_graph.edge_attr  # Raw MLP output (no residual yet)

        # Update world edge features if present (raw MLP output)
        world_edge_mlp_out = None
        if self.use_world_edges and world_edge_attr is not None and world_edge_attr.shape[0] > 0:
            world_graph = Data(
                x=x_input,
                edge_attr=world_edge_attr,
                edge_index=world_edge_index
            )
            world_graph = self.world_eb_module(world_graph)
            world_edge_mlp_out = world_graph.edge_attr  # Raw MLP output (no residual yet)

        # Node update uses RAW edge MLP output (matches DeepMind: residuals applied after)
        node_graph = Data(
            x=x_input,
            edge_attr=edge_mlp_out,
            edge_index=mesh_graph.edge_index
        )
        if self.use_world_edges:
            node_graph.world_edge_attr = world_edge_mlp_out if world_edge_mlp_out is not None else torch.zeros(0, edge_mlp_out.shape[1], device=x_input.device)
            node_graph.world_edge_index = world_edge_index if world_edge_index is not None else torch.zeros(2, 0, dtype=torch.long, device=x_input.device)

        node_graph = self.nb_module(node_graph)

        # Apply all residuals AFTER node update (matches DeepMind original)
        x = x_input + self.residual_scale * node_graph.x

        # PairNorm: center and rescale node features AFTER residual to prevent over-smoothing
        # Applied between layers (not on aggregation) to avoid scale mismatch at depth
        if self.use_pairnorm:
            x_centered = x - x.mean(dim=0, keepdim=True)
            rms = (x_centered.norm(p=2) / (x.shape[0] ** 0.5)) + 1e-8
            x = x_centered / rms

        edge_attr = graph.edge_attr + self.residual_scale * edge_mlp_out  # Edge residual
        updated_world_edge_attr = (world_edge_attr + self.residual_scale * world_edge_mlp_out) if world_edge_mlp_out is not None else world_edge_attr

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
