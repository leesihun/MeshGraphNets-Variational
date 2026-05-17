import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from general_modules.edge_features import EDGE_FEATURE_DIM
from model.checkpointing import process_with_checkpointing
from model.coarsening import pool_features, unpool_features
from model.encoder_decoder import Decoder, Encoder, GnBlock
from model.mlp import build_mlp, init_weights
from model.vae import GNNVariationalEncoder


class MeshGraphNets(nn.Module):
    def __init__(self, config, device: str):
        super().__init__()
        self.config = config
        self.device = device

        self.model = EncoderProcessorDecoder(config).to(device)
        self.model.apply(init_weights)

        # Scale decoder's last layer for better initial predictions.
        # T>1 (delta prediction): scale to ~0 ("predict no change" prior)
        num_timesteps = config.get('num_timesteps', None)
        if num_timesteps is None or num_timesteps > 1:
            with torch.no_grad():
                last_layer = self.model.decoder.decode_module[-1]
                last_layer.weight.mul_(0.01)

        print('MeshGraphNets model created successfully')

    def set_checkpointing(self, enabled: bool):
        self.model.set_checkpointing(enabled)

    def forward(self, graph, debug=False, add_noise=None, use_posterior=None, fixed_z=None):
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
        if add_noise is None:
            add_noise = self.training

        if add_noise:
            noise_std = self.config.get('std_noise', 0.0)
            if noise_std > 0:
                output_var = self.config['output_var']
                noise = torch.randn(graph.x.shape[0], output_var,
                                    device=graph.x.device, dtype=graph.x.dtype) * noise_std
                noise_padded = torch.zeros_like(graph.x)
                noise_padded[:, :output_var] = noise
                graph.x = graph.x + noise_padded
                noise_gamma = self.config.get('noise_gamma', 0.1)
                noise_std_ratio = self.config.get('noise_std_ratio', None)
                if noise_std_ratio is not None:
                    ratio = torch.tensor(noise_std_ratio, device=graph.x.device, dtype=graph.x.dtype)
                    graph.y = graph.y - noise_gamma * noise * ratio
                graph.edge_attr = graph.edge_attr + torch.randn_like(graph.edge_attr) * noise_std

        predicted, vae_losses, aux_loss = self.model(
            graph, debug=debug, use_posterior=use_posterior, fixed_z=fixed_z,
        )
        return predicted, graph.y, vae_losses, aux_loss


class EncoderProcessorDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.message_passing_num = config['message_passing_num']
        self.edge_input_size = int(config['edge_var'])
        if self.edge_input_size != EDGE_FEATURE_DIM:
            raise ValueError(f"edge_var must be {EDGE_FEATURE_DIM}, got {self.edge_input_size}")
        self.latent_dim = config['latent_dim']
        self.use_checkpointing = config.get('use_checkpointing', False)
        self.use_world_edges = config.get('use_world_edges', False)
        self.use_multiscale = config.get('use_multiscale', False)
        self.use_coarse_world_edges = (
            bool(config.get('coarse_world_edges', False))
            and self.use_world_edges
            and self.use_multiscale
        )

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

        self.node_output_size = config['output_var']

        self.encoder = Encoder(
            self.edge_input_size, self.node_input_size, self.latent_dim,
            use_world_edges=self.use_world_edges
        )

        if not self.use_multiscale:
            self.processer_list = nn.ModuleList([
                GnBlock(config, self.latent_dim, use_world_edges=self.use_world_edges)
                for _ in range(self.message_passing_num)
            ])
        else:
            self._build_multiscale_processor(config)

        self.decoder = Decoder(self.latent_dim, self.node_output_size)

        self.use_vae = config.get('use_vae', False)
        if self.use_vae:
            self._build_vae_components(config)

    def _build_multiscale_processor(self, config):
        L = int(config.get('multiscale_levels', 1))
        self.multiscale_levels = L

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
        self.mp_per_level = mp_per_level

        expected_len = 2 * L + 1
        if len(mp_per_level) != expected_len:
            raise ValueError(
                f"mp_per_level must have {expected_len} entries for {L} levels, "
                f"got {len(mp_per_level)}: {mp_per_level}"
            )

        parts = []
        for i in range(L):
            parts.append(f"pre[{i}]={mp_per_level[i]}")
        parts.append(f"coarsest={mp_per_level[L]}")
        for i in range(L - 1, -1, -1):
            parts.append(f"post[{i}]={mp_per_level[2 * L - i]}")
        print(f"  Multiscale V-cycle ({L} levels): {', '.join(parts)}")

        coarse_config = dict(config)
        coarse_config['use_world_edges'] = False

        self.pre_blocks = nn.ModuleList()
        self.post_blocks = nn.ModuleList()
        self.coarse_eb_encoders = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for i in range(L):
            pre_count = mp_per_level[i]
            post_count = mp_per_level[2 * L - i]
            use_we = self.use_world_edges if (i == 0 or self.use_coarse_world_edges) else False
            cfg = config if use_we else coarse_config

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

        self.bipartite_unpool = config.get('bipartite_unpool', False)
        if self.bipartite_unpool:
            from model.blocks import UnpoolBlock
            self.unpool_blocks = nn.ModuleList([
                UnpoolBlock(self.latent_dim, build_mlp) for _ in range(L)
            ])

        coarsest_count = mp_per_level[L]
        coarsest_cfg = config if self.use_coarse_world_edges else coarse_config
        self.coarsest_blocks = nn.ModuleList([
            GnBlock(coarsest_cfg, self.latent_dim, use_world_edges=self.use_coarse_world_edges)
            for _ in range(coarsest_count)
        ])

    def _build_vae_components(self, config):
        self.vae_latent_dim = int(config.get('vae_latent_dim', 32))
        vae_mp_layers = int(config.get('vae_mp_layers', 5))
        self.vae_graph_aware = bool(config.get('vae_graph_aware', False))
        self.vae_free_bits = float(config.get('free_bits', 0.0))
        self.vae_encoder = GNNVariationalEncoder(
            self.node_output_size, self.edge_input_size,
            self.latent_dim, self.vae_latent_dim, num_mp_layers=vae_mp_layers,
            node_input_size=self.node_input_size,
            graph_aware=self.vae_graph_aware,
        )
        if self.vae_graph_aware:
            print(f"  VAE encoder: graph-aware (x [N,{self.node_input_size}] fused with y [N,{self.node_output_size}])")
        if self.vae_free_bits > 0:
            print(f"  VAE free-bits floor: {self.vae_free_bits} nats/dim")

        if not self.use_multiscale:
            self.z_fusers = nn.ModuleList([
                nn.Linear(self.latent_dim + self.vae_latent_dim, self.latent_dim)
                for _ in range(self.message_passing_num)
            ])
        else:
            L = self.multiscale_levels
            self.ms_z_fusers_pre = nn.ModuleList()
            self.ms_z_fusers_post = nn.ModuleList()
            for i in range(L):
                pre_count = self.mp_per_level[i]
                post_count = self.mp_per_level[2 * L - i]
                self.ms_z_fusers_pre.append(nn.ModuleList([
                    nn.Linear(self.latent_dim + self.vae_latent_dim, self.latent_dim)
                    for _ in range(pre_count)
                ]))
                self.ms_z_fusers_post.append(nn.ModuleList([
                    nn.Linear(self.latent_dim + self.vae_latent_dim, self.latent_dim)
                    for _ in range(post_count)
                ]))
            coarsest_count = self.mp_per_level[L]
            self.ms_z_fusers_coarsest = nn.ModuleList([
                nn.Linear(self.latent_dim + self.vae_latent_dim, self.latent_dim)
                for _ in range(coarsest_count)
            ])

        self.aux_decoder = build_mlp(
            self.vae_latent_dim, self.latent_dim,
            2 * self.node_output_size,
            layer_norm=False
        )
        print(f"  VAE: ENABLED (z_dim={self.vae_latent_dim}, vae_mp_layers={vae_mp_layers})")

    # ── VAE helpers ──────────────────────────────────────────────────────────

    def _fuse_z(self, x, z_per_node, fuse_layer):
        return fuse_layer(torch.cat([x, z_per_node], dim=-1))

    def _encode_vae(self, original_y, original_x, original_edge_index, original_edge_attr,
                    original_batch, N, device, dtype, use_posterior, fixed_z=None):
        zero = torch.zeros((), device=device, dtype=torch.float32)
        empty_losses = {'mmd': zero, 'kl': zero, 'kl_raw': zero}
        if fixed_z is not None:
            return fixed_z.to(device=device, dtype=dtype), empty_losses
        if use_posterior and original_y is not None:
            batch = (original_batch if original_batch is not None
                     else torch.zeros(N, dtype=torch.long, device=device))
            z, mu, logvar = self.vae_encoder(
                original_y, original_edge_index, original_edge_attr, batch,
                x=(original_x if self.vae_graph_aware else None),
            )
            mmd = GNNVariationalEncoder.mmd_loss(z.float())
            kl_clamped, kl_raw = GNNVariationalEncoder.kl_loss(
                mu.float(), logvar.float(), free_bits=self.vae_free_bits,
            )
            return z, {'mmd': mmd, 'kl': kl_clamped, 'kl_raw': kl_raw}
        B = int(original_batch.max().item()) + 1 if original_batch is not None else 1
        z = torch.randn(B, self.vae_latent_dim, device=device, dtype=dtype)
        return z, empty_losses

    def _aux_loss(self, z, original_y, original_batch, N, device):
        batch = (original_batch if original_batch is not None
                 else torch.zeros(N, dtype=torch.long, device=device))
        B = z.shape[0]
        y_mean = scatter(original_y, batch, dim=0, dim_size=B, reduce='mean')
        y_centered = original_y - y_mean[batch]
        y_std = scatter(y_centered.pow(2), batch, dim=0, dim_size=B, reduce='mean').sqrt()
        aux_target = torch.cat([y_mean, y_std], dim=-1)
        return torch.nn.functional.mse_loss(self.aux_decoder(z), aux_target)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, graph, debug=False, use_posterior=None, fixed_z=None):
        if use_posterior is None:
            use_posterior = self.training and self.use_vae

        if not self.use_multiscale:
            return self._forward_flat(graph, debug, use_posterior, fixed_z)
        return self._forward_multiscale(graph, debug, use_posterior, fixed_z)

    def _forward_flat(self, graph, debug, use_posterior, fixed_z):
        original_y = getattr(graph, 'y', None)
        original_x = graph.x
        original_batch = getattr(graph, 'batch', None)
        original_edge_attr = graph.edge_attr
        original_edge_index = graph.edge_index

        graph = self.encoder(graph)
        if debug:
            print(f"  After Encoder: x std={graph.x.std().item():.4f}, mean={graph.x.mean().item():.4f}")

        vae_losses = {
            'mmd': torch.zeros((), device=graph.x.device, dtype=torch.float32),
            'kl': torch.zeros((), device=graph.x.device, dtype=torch.float32),
            'kl_raw': torch.zeros((), device=graph.x.device, dtype=torch.float32),
        }
        aux_loss = 0.0
        z_per_node = None
        if self.use_vae:
            N = graph.x.shape[0]
            device = graph.x.device
            dtype = graph.x.dtype
            batch_bc = (original_batch if original_batch is not None
                        else torch.zeros(N, dtype=torch.long, device=device))
            z, vae_losses = self._encode_vae(
                original_y, original_x, original_edge_index, original_edge_attr,
                original_batch, N, device, dtype, use_posterior, fixed_z=fixed_z
            )
            z_per_node = z[batch_bc]
            if self.training and original_y is not None:
                aux_loss = self._aux_loss(z, original_y, original_batch, N, device)

        if self.use_checkpointing and self.training:
            graph = process_with_checkpointing(
                self.processer_list, graph,
                z_fusers=self.z_fusers if self.use_vae else None,
                z_per_node=z_per_node
            )
        else:
            for i, model in enumerate(self.processer_list):
                if self.use_vae and z_per_node is not None:
                    graph.x = self._fuse_z(graph.x, z_per_node, self.z_fusers[i])
                graph = model(graph)
                if debug and i == len(self.processer_list) - 1:
                    print(f"  After MP block {i}: x std={graph.x.std().item():.4f}, mean={graph.x.mean().item():.4f}")

        output = self.decoder(graph)
        if debug:
            print(f"  After Decoder: out std={output.std().item():.4f}, mean={output.mean().item():.4f}")
        return output, vae_losses, aux_loss

    def _forward_multiscale(self, graph, debug, use_posterior, fixed_z):
        L = self.multiscale_levels

        original_y = getattr(graph, 'y', None)
        original_x = graph.x
        original_batch = getattr(graph, 'batch', None)
        original_edge_attr = graph.edge_attr
        original_edge_index = graph.edge_index

        level_data = self._extract_level_data(graph, L)
        actual_levels = len(level_data)

        graph = self.encoder(graph)
        if debug:
            print(f"  [MS] After Encoder: x std={graph.x.std().item():.4f}")

        vae_losses = {
            'mmd': torch.zeros((), device=graph.x.device, dtype=torch.float32),
            'kl': torch.zeros((), device=graph.x.device, dtype=torch.float32),
            'kl_raw': torch.zeros((), device=graph.x.device, dtype=torch.float32),
        }
        aux_loss = 0.0
        z = None
        current_z_per_node = None
        current_batch = None
        if self.use_vae:
            N = graph.x.shape[0]
            device = graph.x.device
            dtype = graph.x.dtype
            batch_bc = (original_batch if original_batch is not None
                        else torch.zeros(N, dtype=torch.long, device=device))
            z, vae_losses = self._encode_vae(
                original_y, original_x, original_edge_index, original_edge_attr,
                original_batch, N, device, dtype, use_posterior, fixed_z=fixed_z
            )
            current_z_per_node = z[batch_bc]
            current_batch = batch_bc
            if self.training and original_y is not None:
                aux_loss = self._aux_loss(z, original_y, original_batch, N, device)

        # Descending arm (fine → coarse)
        skip_states = []
        current_graph = graph

        for i in range(actual_levels):
            z_fusers_pre = self.ms_z_fusers_pre[i] if (self.use_vae and current_z_per_node is not None) else None
            current_graph = self._run_processor_blocks(
                self.pre_blocks[i], current_graph, z_fusers_pre, current_z_per_node
            )
            if debug:
                print(f"  [MS] After pre[{i}] ({len(self.pre_blocks[i])} blocks): x std={current_graph.x.std().item():.4f}")

            use_we_here = self.use_world_edges and (i == 0 or self.use_coarse_world_edges)
            skip_states.append({
                'x': current_graph.x,
                'edge_attr': current_graph.edge_attr,
                'edge_index': current_graph.edge_index,
                'w_attr': getattr(current_graph, 'world_edge_attr', None) if use_we_here else None,
                'w_idx': getattr(current_graph, 'world_edge_index', None) if use_we_here else None,
                'z_per_node': current_z_per_node,
            })

            ld = level_data[i]
            h_coarse = pool_features(current_graph.x, ld['ftc'], ld['n_c'])
            e_coarse = self.coarse_eb_encoders[i](ld['c_ea'])
            current_graph = Data(x=h_coarse, edge_attr=e_coarse, edge_index=ld['c_ei'])
            if self.use_coarse_world_edges and ld['c_we_idx'] is not None and ld['c_we_idx'].shape[1] > 0:
                current_graph.world_edge_attr  = ld['c_we_attr']
                current_graph.world_edge_index = ld['c_we_idx']

            if self.use_vae and z is not None:
                current_batch = scatter(current_batch, ld['ftc'], dim=0, dim_size=ld['n_c'], reduce='min')
                current_z_per_node = z[current_batch]

            if debug:
                print(f"  [MS] After pool[{i}]: {skip_states[-1]['x'].shape[0]} → {h_coarse.shape[0]} nodes")

        # Coarsest level
        z_fusers_coarsest = self.ms_z_fusers_coarsest if (self.use_vae and current_z_per_node is not None) else None
        current_graph = self._run_processor_blocks(
            self.coarsest_blocks, current_graph, z_fusers_coarsest, current_z_per_node
        )
        if debug:
            print(f"  [MS] After coarsest ({len(self.coarsest_blocks)} blocks): x std={current_graph.x.std().item():.4f}")

        # Ascending arm (coarse → fine)
        for i in range(actual_levels - 1, -1, -1):
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

            skip = skip_states[i]
            h_merged = self.skip_projs[i](torch.cat([skip['x'], h_up], dim=-1))
            current_graph = Data(x=h_merged, edge_attr=skip['edge_attr'], edge_index=skip['edge_index'])
            use_we_here = self.use_world_edges and (i == 0 or self.use_coarse_world_edges)
            if use_we_here and skip['w_attr'] is not None:
                current_graph.world_edge_attr  = skip['w_attr']
                current_graph.world_edge_index = skip['w_idx']

            level_z_per_node = skip_states[i].get('z_per_node') if self.use_vae else None
            z_fusers_post = self.ms_z_fusers_post[i] if (self.use_vae and level_z_per_node is not None) else None
            current_graph = self._run_processor_blocks(
                self.post_blocks[i], current_graph, z_fusers_post, level_z_per_node
            )
            if debug:
                print(f"  [MS] After post[{i}] ({len(self.post_blocks[i])} blocks): x std={current_graph.x.std().item():.4f}")

        output = self.decoder(current_graph)
        if debug:
            print(f"  [MS] After Decoder: out std={output.std().item():.4f}")
        return output, vae_losses, aux_loss

    def _extract_level_data(self, graph, L):
        """Extract per-level coarsening topology from graph before encoder drops custom attrs."""
        level_data = {}
        for i in range(L):
            ftc_key = f'fine_to_coarse_{i}'
            if not hasattr(graph, ftc_key):
                break
            ld = {
                'ftc': graph[ftc_key],
                'c_ei': graph[f'coarse_edge_index_{i}'],
                'c_ea': graph[f'coarse_edge_attr_{i}'],
                'n_c': int(graph[f'num_coarse_{i}'].sum()),
                'c_we_idx': getattr(graph, f'coarse_world_edge_index_{i}', None),
                'c_we_attr': getattr(graph, f'coarse_world_edge_attr_{i}', None),
            }
            if self.use_multiscale and getattr(self, 'bipartite_unpool', False):
                ld['up_ei'] = graph[f'unpool_edge_index_{i}']
                ld['coarse_centroid'] = graph[f'coarse_centroid_{i}']
                ld['fine_pos'] = graph.pos if i == 0 else graph[f'coarse_centroid_{i - 1}']
            level_data[i] = ld
        return level_data

    def _run_processor_blocks(self, blocks, graph, z_fusers, z_per_node):
        """Run a list of GnBlocks with optional per-block z injection."""
        if self.use_checkpointing and self.training:
            return process_with_checkpointing(blocks, graph, z_fusers=z_fusers, z_per_node=z_per_node)
        for j, block in enumerate(blocks):
            if z_fusers is not None and z_per_node is not None:
                graph.x = self._fuse_z(graph.x, z_per_node, z_fusers[j])
            graph = block(graph)
        return graph

    def set_checkpointing(self, enabled: bool):
        self.use_checkpointing = enabled
