"""ModelSplitStage — one pipeline stage of a sliced MeshGraphNets.

Supports:
  - Flat processor  (use_multiscale=False)
  - Multiscale V-cycle (use_multiscale=True)
  - VAE conditioning   (use_vae=True)  — z_per_node is sent *detached* across
    stage boundaries; vae_encoder trains through its local MMD/aux losses only
  - World edges        (use_world_edges=True)

EMA and torch.compile are handled in launcher.py.

Cross-rank wire protocol
------------------------
BundleSend / BundleRecv replace the old 3-tensor SendActivations/RecvActivations.
They carry a variable-length, STATICALLY-SIZED bundle of grad-bearing + non-grad
tensors in a fixed canonical order determined once at stage construction time.

Bundle layout (canonical order, always the same n_grad / n_nongd per boundary):
  Grad tensors:
    [0]  x              [N, D]
    [1]  edge_attr      [E, D]
    [+W] world_edge_attr [E_w, D]  (if use_world_edges)
    for each skip i in 0 .. n_out_skips-1:
      skip_x[i]    [N_i, D]
      skip_ea[i]   [E_i, D]
    [+W] skip_w_attr  [E_w0, D]  (if use_world_edges and n_skips > 0 — level-0 world edges)

  Non-grad tensors:
    [0]  edge_index      [2, E]   int64
    [+W] world_edge_idx  [2, E_w] int64 (if use_world_edges)
    for each skip i:
      skip_ei[i]     [2, E_i]  int64
    [+W] skip_w_idx  [2, E_w0] int64  (if use_world_edges and n_skips > 0)
    for each skip i:
      skip_zpn[i]  [N_i, vae_dim]  float, detached  (if use_vae)
    [+V] z_per_node_cur  [N, vae_dim]  float, detached  (if use_vae)
    [+1] meta  [2] = [current_level_idx, n_skips]  int64  (if use_multiscale)

where W = 1 if use_world_edges, V = 1 if use_vae.

NCCL note: send/recv pairs are matched strictly by ORDER (NCCL ignores tag).
All sends/recvs are inside ONE autograd Function per boundary so order is
deterministic on both ranks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from general_modules.edge_features import EDGE_FEATURE_DIM
from model.coarsening import pool_features, unpool_features
from model.encoder_decoder import Decoder, Encoder, GnBlock
from model.mlp import build_mlp
from model.vae import GNNVariationalEncoder


# ---------------------------------------------------------------------------
# Wire-level helpers
# ---------------------------------------------------------------------------

_DTYPE_TO_CODE = {
    torch.float32: 0, torch.float64: 1, torch.bfloat16: 2, torch.float16: 3,
    torch.int64: 4, torch.int32: 5, torch.bool: 6,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


def _send_tensor(t: torch.Tensor, dst: int) -> None:
    dtype_code = _DTYPE_TO_CODE[t.dtype]
    shape = list(t.shape)
    header = torch.tensor([len(shape), *shape, dtype_code], dtype=torch.long, device=t.device)
    header_len = torch.tensor([header.numel()], dtype=torch.long, device=t.device)
    dist.send(header_len, dst=dst)
    dist.send(header, dst=dst)
    dist.send(t.contiguous(), dst=dst)


def _recv_tensor(src: int, device: torch.device) -> torch.Tensor:
    header_len = torch.empty(1, dtype=torch.long, device=device)
    dist.recv(header_len, src=src)
    header = torch.empty(int(header_len.item()), dtype=torch.long, device=device)
    dist.recv(header, src=src)
    ndim = int(header[0].item())
    shape = [int(header[1 + i].item()) for i in range(ndim)]
    dtype = _CODE_TO_DTYPE[int(header[1 + ndim].item())]
    t = torch.empty(*shape, dtype=dtype, device=device)
    dist.recv(t, src=src)
    return t


# ---------------------------------------------------------------------------
# BundleSend / BundleRecv — variadic autograd bridges
# ---------------------------------------------------------------------------

class BundleSend(torch.autograd.Function):
    """Send a bundle of (grad_bearing | non_differentiable) tensors downstream.

    Signature: BundleSend.apply(dst, n_grad, *all_tensors)
      dst      — int, destination rank
      n_grad   — int, first n_grad tensors carry gradients; rest are non-grad
      *all_tensors — flat tuple of tensors in canonical order
    """

    @staticmethod
    def forward(ctx, dst: int, n_grad: int, *all_tensors: torch.Tensor) -> torch.Tensor:
        ctx.dst = dst
        ctx.n_grad = n_grad
        ctx.n_nongd = len(all_tensors) - n_grad
        ctx.device = all_tensors[0].device
        if ctx.n_nongd > 0:
            ctx.mark_non_differentiable(*all_tensors[n_grad:])
        for t in all_tensors:
            _send_tensor(t.contiguous(), dst=dst)
        return torch.zeros((), device=ctx.device, dtype=all_tensors[0].dtype,
                           requires_grad=True)

    @staticmethod
    def backward(ctx, _grad_sentinel):
        grads = [_recv_tensor(src=ctx.dst, device=ctx.device)
                 for _ in range(ctx.n_grad)]
        return (None, None) + tuple(grads) + (None,) * ctx.n_nongd


class BundleRecv(torch.autograd.Function):
    """Receive a bundle from upstream; send grads upstream in backward.

    Signature: BundleRecv.apply(src, n_grad, n_nongd, device, anchor)
    Returns a tuple of tensors (n_grad + n_nongd elements).
    """

    @staticmethod
    def forward(ctx, src: int, n_grad: int, n_nongd: int,
                device: torch.device, anchor: torch.Tensor):
        ctx.src = src
        ctx.n_grad = n_grad
        tensors = [_recv_tensor(src=src, device=device)
                   for _ in range(n_grad + n_nongd)]
        if n_nongd > 0:
            ctx.mark_non_differentiable(*tensors[n_grad:])
        return tuple(tensors)

    @staticmethod
    def backward(ctx, *grads):
        for g in grads[:ctx.n_grad]:
            _send_tensor(g.contiguous(), dst=ctx.src)
        return (None, None, None, None, None)


# ---------------------------------------------------------------------------
# Bundle spec helpers
# ---------------------------------------------------------------------------

def _bundle_counts(use_world_edges: bool, use_vae: bool,
                   n_skips: int, is_multiscale: bool,
                   use_coarse_world_edges: bool = False) -> Tuple[int, int]:
    """Return (n_grad, n_nongd) for a bundle with n_skips accumulated skip states.

    When use_coarse_world_edges=True every skip carries its own world-edge pair
    (w_attr in grad, w_idx in nongd). When False (default) only skip[0] does.
    """
    W  = 1 if use_world_edges else 0
    V  = 1 if use_vae else 0
    ms = 1 if is_multiscale else 0

    if use_coarse_world_edges:
        # Every skip: (x, ea, w_attr) in grad; (ei, w_idx) in nongd
        n_grad  = 2 + W + (2 + W) * n_skips
        n_nongd = 1 + W + (1 + W) * n_skips + n_skips * V + V + ms
    else:
        # Only skip[0] carries world edges (original layout)
        n_grad  = 2 + W + 2 * n_skips + (W if n_skips > 0 else 0)
        n_nongd = (1 + W + n_skips + (W if n_skips > 0 else 0)
                   + n_skips * V + V + ms)
    return n_grad, n_nongd


def _pack_bundle(
    x: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_index: torch.Tensor,
    skip_stack: List[dict],
    world_edge_attr: Optional[torch.Tensor],
    world_edge_index: Optional[torch.Tensor],
    z_per_node_cur: Optional[torch.Tensor],
    current_level_idx: int,
    use_world_edges: bool,
    use_vae: bool,
    is_multiscale: bool,
    use_coarse_world_edges: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Pack state into (grad_tensors, nongd_tensors) in canonical order."""
    dev = x.device
    D = x.shape[-1]
    V = z_per_node_cur.shape[-1] if (use_vae and z_per_node_cur is not None) else 0

    grad_t: List[torch.Tensor] = [x, edge_attr]
    nongd_t: List[torch.Tensor] = [edge_index]

    if use_world_edges:
        wea = world_edge_attr if world_edge_attr is not None else torch.zeros(0, D, device=dev, dtype=x.dtype)
        wei = world_edge_index if world_edge_index is not None else torch.zeros(2, 0, dtype=torch.long, device=dev)
        grad_t.append(wea)
        nongd_t.append(wei)

    # Skip states: (x, ea) in grad; ei in nongd
    for ss in skip_stack:
        grad_t.extend([ss['x'], ss['edge_attr']])
        nongd_t.append(ss['edge_index'])

    # World edges for skips: all skips when use_coarse_world_edges, else only skip[0]
    if use_world_edges and len(skip_stack) > 0:
        if use_coarse_world_edges:
            for ss in skip_stack:
                w = ss.get('w_attr')
                grad_t.append(w if w is not None else torch.zeros(0, D, device=dev, dtype=x.dtype))
            for ss in skip_stack:
                wi = ss.get('w_idx')
                nongd_t.append(wi if wi is not None else torch.zeros(2, 0, dtype=torch.long, device=dev))
        else:
            sw_attr = skip_stack[0].get('w_attr')
            sw_idx  = skip_stack[0].get('w_idx')
            grad_t.append(sw_attr if sw_attr is not None else torch.zeros(0, D, device=dev, dtype=x.dtype))
            nongd_t.append(sw_idx  if sw_idx  is not None else torch.zeros(2, 0, dtype=torch.long, device=dev))

    # Per-skip z_per_node
    if use_vae:
        for ss in skip_stack:
            zpn = ss.get('z_per_node')
            if zpn is not None:
                nongd_t.append(zpn.detach())
            else:
                N_i = ss['x'].shape[0]
                nongd_t.append(torch.zeros(N_i, V, device=dev, dtype=x.dtype))

    # Current z_per_node
    if use_vae:
        if z_per_node_cur is not None:
            nongd_t.append(z_per_node_cur.detach())
        else:
            nongd_t.append(torch.zeros(x.shape[0], V, device=dev, dtype=x.dtype))

    # Meta
    if is_multiscale:
        nongd_t.append(torch.tensor([current_level_idx, len(skip_stack)],
                                    dtype=torch.long, device=dev))

    return grad_t, nongd_t



def _unpack_bundle_indexed(
    all_tensors: tuple,
    n_skips: int,
    use_world_edges: bool,
    use_vae: bool,
    is_multiscale: bool,
    use_coarse_world_edges: bool = False,
) -> Tuple:
    """Unpack using canonical index positions. Returns:
       (x, ea, ei, world_ea, world_ei, skip_stack, z_per_node_cur, current_level_idx)
    """
    W = 1 if use_world_edges else 0
    idx = 0

    # --- Grad section ---
    x = all_tensors[idx]; idx += 1
    edge_attr = all_tensors[idx]; idx += 1
    world_edge_attr = all_tensors[idx] if use_world_edges else None
    if use_world_edges: idx += 1

    skip_x_list, skip_ea_list = [], []
    for _ in range(n_skips):
        skip_x_list.append(all_tensors[idx]); idx += 1
        skip_ea_list.append(all_tensors[idx]); idx += 1

    # World attrs for skips: all skips if use_coarse_world_edges, else only skip[0]
    skip_w_attr_list: List[Optional[torch.Tensor]] = [None] * n_skips
    if use_world_edges and n_skips > 0:
        if use_coarse_world_edges:
            for i in range(n_skips):
                skip_w_attr_list[i] = all_tensors[idx]; idx += 1
        else:
            skip_w_attr_list[0] = all_tensors[idx]; idx += 1

    # --- Non-grad section ---
    n_grad, _ = _bundle_counts(use_world_edges, use_vae, n_skips, is_multiscale,
                                use_coarse_world_edges)
    idx = n_grad  # reset to start of nongd section

    edge_index = all_tensors[idx]; idx += 1
    world_edge_index = all_tensors[idx] if use_world_edges else None
    if use_world_edges: idx += 1

    skip_ei_list = [all_tensors[idx + i] for i in range(n_skips)]
    idx += n_skips

    # World indices for skips: all skips if use_coarse_world_edges, else only skip[0]
    skip_w_idx_list: List[Optional[torch.Tensor]] = [None] * n_skips
    if use_world_edges and n_skips > 0:
        if use_coarse_world_edges:
            for i in range(n_skips):
                skip_w_idx_list[i] = all_tensors[idx]; idx += 1
        else:
            skip_w_idx_list[0] = all_tensors[idx]; idx += 1

    skip_zpn_list = []
    if use_vae:
        for _ in range(n_skips):
            skip_zpn_list.append(all_tensors[idx]); idx += 1

    z_per_node_cur = None
    if use_vae:
        z_per_node_cur = all_tensors[idx]; idx += 1

    current_level_idx = 0
    if is_multiscale:
        meta = all_tensors[idx]; idx += 1
        current_level_idx = int(meta[0].item())

    # Build skip_stack
    skip_stack = []
    for i in range(n_skips):
        ss = {
            'x': skip_x_list[i],
            'edge_attr': skip_ea_list[i],
            'edge_index': skip_ei_list[i],
        }
        ss['w_attr'] = skip_w_attr_list[i]
        ss['w_idx']  = skip_w_idx_list[i]
        if use_vae and skip_zpn_list:
            ss['z_per_node'] = skip_zpn_list[i]
        else:
            ss['z_per_node'] = None
        skip_stack.append(ss)

    return (x, edge_attr, edge_index,
            world_edge_attr, world_edge_index,
            skip_stack, z_per_node_cur, current_level_idx)


# ---------------------------------------------------------------------------
# V-cycle helpers (multiscale)
# ---------------------------------------------------------------------------

def _parse_mp_per_level(config: dict, L: int) -> List[int]:
    mp = config.get('mp_per_level', None)
    if mp is None:
        fine_pre  = int(config.get('fine_mp_pre', 5))
        coarse_mp = int(config.get('coarse_mp_num', 5))
        fine_post = int(config.get('fine_mp_post', 5))
        mp = [fine_pre] + [coarse_mp] + [fine_post]
    if not isinstance(mp, list):
        mp = [int(mp)]
    else:
        mp = [int(x) for x in mp]
    return mp


def _block_vcycle_info(b: int, L: int, mp_per_level: List[int]) -> Tuple[str, Optional[int], int]:
    """Map flat block index b (0-based) to (kind, level, local_idx)."""
    cumulative = 0
    for i in range(L):
        c = mp_per_level[i]
        if b < cumulative + c:
            return 'pre', i, b - cumulative
        cumulative += c
    c = mp_per_level[L]
    if b < cumulative + c:
        return 'coarsest', None, b - cumulative
    cumulative += c
    for i in range(L - 1, -1, -1):
        c = mp_per_level[2 * L - i]
        if b < cumulative + c:
            return 'post', i, b - cumulative
        cumulative += c
    raise IndexError(f"block index {b} out of range for L={L}, mp_per_level={mp_per_level}")


def _build_stage_ops(my_block_indices: List[int], L: int,
                     mp_per_level: List[int]) -> List[tuple]:
    """Build execution ops list for this stage's blocks.

    Each op is one of:
      ('block',     kind, level, local_idx)
      ('save_pool', level)   — save skip state then pool to next level
      ('unpool',    level)   — unpool + merge skip, pop stack
    """
    ops = []
    for b in my_block_indices:
        kind, level, local_idx = _block_vcycle_info(b, L, mp_per_level)
        if kind == 'post' and local_idx == 0:
            ops.append(('unpool', level))
        ops.append(('block', kind, level, local_idx))
        if kind == 'pre' and local_idx == mp_per_level[level] - 1:
            ops.append(('save_pool', level))
    return ops


def _compute_out_skip_depth(my_block_indices: List[int], L: int,
                             mp_per_level: List[int], in_skip_depth: int) -> int:
    depth = in_skip_depth
    for b in my_block_indices:
        kind, level, local_idx = _block_vcycle_info(b, L, mp_per_level)
        if kind == 'post' and local_idx == 0:
            depth -= 1
        if kind == 'pre' and local_idx == mp_per_level[level] - 1:
            depth += 1
    return depth


def _compute_in_skip_depth(my_block_indices: List[int], L: int,
                            mp_per_level: List[int]) -> int:
    """Compute the skip depth this stage receives from its predecessor."""
    if not my_block_indices:
        return 0
    first_b = my_block_indices[0]
    depth = 0
    # Simulate from block 0 to first_b-1 to find accumulated depth
    for b in range(first_b):
        kind, level, local_idx = _block_vcycle_info(b, L, mp_per_level)
        if kind == 'post' and local_idx == 0:
            depth -= 1
        if kind == 'pre' and local_idx == mp_per_level[level] - 1:
            depth += 1
    return depth


# ---------------------------------------------------------------------------
# Stage inner module
# ---------------------------------------------------------------------------

class _StageInner(nn.Module):
    """Holds all learnable parameters for one pipeline stage.

    State-dict keys mirror those of the full single-GPU EncoderProcessorDecoder
    so that merged checkpoints can be loaded directly into the full model.
    """

    def __init__(
        self,
        *,
        is_first: bool,
        is_last: bool,
        ops_sequence: List[tuple],   # from _build_stage_ops
        my_blocks: List[int],
        config: dict,
        edge_input_size: int,
        node_input_size: int,
        node_output_size: int,
        latent_dim: int,
        use_world_edges: bool,
        use_vae: bool,
        use_multiscale: bool,
        L: int = 0,
        mp_per_level: Optional[List[int]] = None,
    ):
        super().__init__()
        coarse_config = dict(config)
        coarse_config['use_world_edges'] = False

        if is_first:
            self.encoder = Encoder(
                edge_input_size, node_input_size, latent_dim,
                use_world_edges=use_world_edges,
            )

        if is_last:
            self.decoder = Decoder(latent_dim, node_output_size)

        # ---- Processor blocks ----
        if not use_multiscale:
            self.processer_list = nn.ModuleDict()
            for i in my_blocks:
                self.processer_list[str(i)] = GnBlock(
                    config, latent_dim, use_world_edges=use_world_edges)
        else:
            assert L > 0 and mp_per_level is not None
            self._build_multiscale_blocks(
                ops_sequence, config, coarse_config, latent_dim,
                edge_input_size, use_world_edges, L, mp_per_level,
            )

        # ---- VAE components ----
        if use_vae:
            self._build_vae_components(
                config, is_first, my_blocks, latent_dim, node_output_size,
                node_input_size, edge_input_size, use_multiscale, L, mp_per_level,
                ops_sequence,
            )

    # ------------------------------------------------------------------
    def _build_multiscale_blocks(self, ops_sequence, config, coarse_config,
                                  latent_dim, edge_input_size, use_world_edges,
                                  L, mp_per_level):
        pre_dict: Dict[str, Dict[str, nn.Module]] = {}
        post_dict: Dict[str, Dict[str, nn.Module]] = {}
        coarsest_dict: Dict[str, nn.Module] = {}
        coarse_eb_dict: Dict[str, nn.Module] = {}
        skip_proj_dict: Dict[str, nn.Module] = {}
        unpool_dict: Dict[str, nn.Module] = {}

        bipartite_unpool = bool(config.get('bipartite_unpool', False))
        use_coarse_we = bool(config.get('coarse_world_edges', False)) and use_world_edges

        for op in ops_sequence:
            if op[0] == 'block':
                _, kind, level, local_idx = op
                if kind == 'pre':
                    lv = str(level)
                    li = str(local_idx)
                    if lv not in pre_dict:
                        pre_dict[lv] = {}
                    use_we = use_world_edges if (level == 0 or use_coarse_we) else False
                    cfg = config if use_we else coarse_config
                    if li not in pre_dict[lv]:
                        pre_dict[lv][li] = GnBlock(cfg, latent_dim, use_world_edges=use_we)
                elif kind == 'coarsest':
                    li = str(local_idx)
                    if li not in coarsest_dict:
                        coarsest_dict[li] = GnBlock(
                            config if use_coarse_we else coarse_config,
                            latent_dim,
                            use_world_edges=use_coarse_we,
                        )
                elif kind == 'post':
                    lv = str(level)
                    li = str(local_idx)
                    if lv not in post_dict:
                        post_dict[lv] = {}
                    use_we = use_world_edges if (level == 0 or use_coarse_we) else False
                    cfg = config if use_we else coarse_config
                    if li not in post_dict[lv]:
                        post_dict[lv][li] = GnBlock(cfg, latent_dim, use_world_edges=use_we)
            elif op[0] == 'save_pool':
                level = op[1]
                lv = str(level)
                if lv not in coarse_eb_dict:
                    coarse_eb_dict[lv] = build_mlp(edge_input_size, latent_dim, latent_dim)
            elif op[0] == 'unpool':
                level = op[1]
                lv = str(level)
                if lv not in skip_proj_dict:
                    skip_proj_dict[lv] = nn.Linear(2 * latent_dim, latent_dim)
                if bipartite_unpool and lv not in unpool_dict:
                    from model.blocks import UnpoolBlock
                    unpool_dict[lv] = UnpoolBlock(latent_dim, build_mlp)

        # Register as nested ModuleDicts matching full model key structure
        if pre_dict:
            self.pre_blocks = nn.ModuleDict({
                lv: nn.ModuleDict(blocks) for lv, blocks in pre_dict.items()
            })
        if coarsest_dict:
            self.coarsest_blocks = nn.ModuleDict(coarsest_dict)
        if post_dict:
            self.post_blocks = nn.ModuleDict({
                lv: nn.ModuleDict(blocks) for lv, blocks in post_dict.items()
            })
        if coarse_eb_dict:
            self.coarse_eb_encoders = nn.ModuleDict(coarse_eb_dict)
        if skip_proj_dict:
            self.skip_projs = nn.ModuleDict(skip_proj_dict)
        if unpool_dict:
            self.unpool_blocks = nn.ModuleDict(unpool_dict)

    # ------------------------------------------------------------------
    def _build_vae_components(self, config, is_first, my_blocks, latent_dim,
                               node_output_size, node_input_size, edge_input_size,
                               use_multiscale, L, mp_per_level, ops_sequence):
        vae_latent_dim = int(config.get('vae_latent_dim', 32))
        if is_first:
            vae_mp_layers = int(config.get('vae_mp_layers', 5))
            vae_graph_aware = bool(config.get('vae_graph_aware', False))
            self.vae_encoder = GNNVariationalEncoder(
                node_output_size, edge_input_size, latent_dim, vae_latent_dim,
                num_mp_layers=vae_mp_layers,
                node_input_size=node_input_size if vae_graph_aware else None,
                graph_aware=vae_graph_aware,
            )
            self.aux_decoder = build_mlp(
                vae_latent_dim, latent_dim, 2 * node_output_size, layer_norm=False)

        if not use_multiscale:
            self.z_fusers = nn.ModuleDict({
                str(i): nn.Linear(latent_dim + vae_latent_dim, latent_dim)
                for i in my_blocks
            })
        else:
            pre_fuser_dict: Dict[str, Dict[str, nn.Module]] = {}
            post_fuser_dict: Dict[str, Dict[str, nn.Module]] = {}
            coarsest_fuser_dict: Dict[str, nn.Module] = {}
            for op in ops_sequence:
                if op[0] == 'block':
                    _, kind, level, local_idx = op
                    lin = nn.Linear(latent_dim + vae_latent_dim, latent_dim)
                    if kind == 'pre':
                        lv, li = str(level), str(local_idx)
                        if lv not in pre_fuser_dict:
                            pre_fuser_dict[lv] = {}
                        pre_fuser_dict[lv][li] = lin
                    elif kind == 'coarsest':
                        li = str(local_idx)
                        coarsest_fuser_dict[li] = lin
                    elif kind == 'post':
                        lv, li = str(level), str(local_idx)
                        if lv not in post_fuser_dict:
                            post_fuser_dict[lv] = {}
                        post_fuser_dict[lv][li] = lin
            if pre_fuser_dict:
                self.ms_z_fusers_pre = nn.ModuleDict({
                    lv: nn.ModuleDict(d) for lv, d in pre_fuser_dict.items()
                })
            if coarsest_fuser_dict:
                self.ms_z_fusers_coarsest = nn.ModuleDict(coarsest_fuser_dict)
            if post_fuser_dict:
                self.ms_z_fusers_post = nn.ModuleDict({
                    lv: nn.ModuleDict(d) for lv, d in post_fuser_dict.items()
                })


# ---------------------------------------------------------------------------
# Stage module
# ---------------------------------------------------------------------------

class ModelSplitStage(nn.Module):
    """One pipeline stage; only holds the parameters assigned to this rank."""

    def __init__(
        self,
        config: dict,
        stage_idx: int,
        num_stages: int,
        assignment: Sequence[Sequence[int]],
        device: torch.device,
    ):
        super().__init__()
        self.config = config
        self.stage_idx = int(stage_idx)
        self.num_stages = int(num_stages)
        self.is_first = (self.stage_idx == 0)
        self.is_last  = (self.stage_idx == self.num_stages - 1)
        self.device   = device

        self.use_multiscale  = bool(config.get('use_multiscale', False))
        self.use_vae         = bool(config.get('use_vae', False))
        self.use_world_edges = bool(config.get('use_world_edges', False))
        self.use_coarse_world_edges = (
            bool(config.get('coarse_world_edges', False))
            and self.use_world_edges and self.use_multiscale
        )

        my_blocks = sorted(assignment[stage_idx])
        self.my_block_indices = my_blocks

        latent_dim     = int(config['latent_dim'])
        edge_input_size = int(config['edge_var'])
        if edge_input_size != EDGE_FEATURE_DIM:
            raise ValueError(f"edge_var must be {EDGE_FEATURE_DIM}, got {edge_input_size}")

        base_input_size = int(config['input_var'])
        base_input_size += int(config.get('positional_features', 0))
        if config.get('use_node_types', False) and int(config.get('num_node_types', 0)) > 0:
            node_input_size = base_input_size + int(config['num_node_types'])
        else:
            node_input_size = base_input_size

        # ---- Multiscale layout ----
        L = 0
        mp_per_level: List[int] = []
        ops_sequence: List[tuple] = []
        self._in_skip_depth  = 0
        self._out_skip_depth = 0

        if self.use_multiscale:
            L = int(config.get('multiscale_levels', 1))
            mp_per_level = _parse_mp_per_level(config, L)
            ops_sequence = _build_stage_ops(my_blocks, L, mp_per_level)
            self._in_skip_depth  = _compute_in_skip_depth(my_blocks, L, mp_per_level)
            self._out_skip_depth = _compute_out_skip_depth(
                my_blocks, L, mp_per_level, self._in_skip_depth)
        self._L = L
        self._mp_per_level = mp_per_level
        self._ops_sequence = ops_sequence

        # ---- VAE dim ----
        self._vae_latent_dim = int(config.get('vae_latent_dim', 32)) if self.use_vae else 0

        # ---- Build inner module ----
        self.model = _StageInner(
            is_first=self.is_first,
            is_last=self.is_last,
            ops_sequence=ops_sequence,
            my_blocks=my_blocks,
            config=config,
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            node_output_size=int(config['output_var']),
            latent_dim=latent_dim,
            use_world_edges=self.use_world_edges,
            use_vae=self.use_vae,
            use_multiscale=self.use_multiscale,
            L=L,
            mp_per_level=mp_per_level if self.use_multiscale else None,
        )

        self.to(device)

        # Decoder last-layer scale
        num_timesteps = config.get('num_timesteps', None)
        if (num_timesteps is None or num_timesteps > 1) and self.is_last:
            with torch.no_grad():
                self.model.decoder.decode_module[-1].weight.mul_(0.01)

    # ------------------------------------------------------------------
    # Send / recv helpers
    # ------------------------------------------------------------------

    def _n_out_skips(self) -> int:
        return self._out_skip_depth

    def _n_in_skips(self) -> int:
        return self._in_skip_depth

    def send_to_next(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        skip_stack: List[dict],
        world_edge_attr: Optional[torch.Tensor],
        world_edge_index: Optional[torch.Tensor],
        z_per_node_cur: Optional[torch.Tensor],
        current_level_idx: int,
    ) -> torch.Tensor:
        grad_t, nongd_t = _pack_bundle(
            x, edge_attr, edge_index, skip_stack,
            world_edge_attr, world_edge_index, z_per_node_cur,
            current_level_idx,
            self.use_world_edges, self.use_vae, self.use_multiscale,
            self.use_coarse_world_edges,
        )
        n_grad, n_nongd = _bundle_counts(
            self.use_world_edges, self.use_vae, len(skip_stack), self.use_multiscale,
            self.use_coarse_world_edges)
        dst = self.stage_idx + 1
        return BundleSend.apply(dst, n_grad, *grad_t, *nongd_t)

    def recv_from_prev(self) -> tuple:
        src = self.stage_idx - 1
        n_skips = self._in_skip_depth
        n_grad, n_nongd = _bundle_counts(
            self.use_world_edges, self.use_vae, n_skips, self.use_multiscale,
            self.use_coarse_world_edges)
        anchor = torch.zeros((), device=self.device, requires_grad=True)
        all_tensors = BundleRecv.apply(src, n_grad, n_nongd, self.device, anchor)
        return _unpack_bundle_indexed(
            all_tensors, n_skips,
            self.use_world_edges, self.use_vae, self.use_multiscale,
            self.use_coarse_world_edges,
        )

    # ------------------------------------------------------------------
    # Stage-0 helpers
    # ------------------------------------------------------------------

    def apply_input_noise(self, graph) -> None:
        noise_std = self.config.get('std_noise', 0.0)
        if noise_std <= 0:
            return
        output_var = int(self.config['output_var'])
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

    def encode(self, graph) -> Tuple:
        """Run encoder. Returns (x, ea, ei, world_ea, world_ei)."""
        if not self.is_first:
            raise RuntimeError("encode() called on non-first stage")
        encoded = self.model.encoder(graph)
        wea = getattr(encoded, 'world_edge_attr', None) if self.use_world_edges else None
        wei = getattr(encoded, 'world_edge_index', None) if self.use_world_edges else None
        return encoded.x, encoded.edge_attr, encoded.edge_index, wea, wei

    def encode_vae(self, graph) -> Tuple:
        """Run VAE encoder on graph (stage 0 only). Returns (z_per_node, vae_losses, aux_loss)."""
        if not (self.is_first and self.use_vae):
            raise RuntimeError("encode_vae() called on wrong stage")
        original_y    = getattr(graph, 'y', None)
        original_x    = graph.x
        original_batch = getattr(graph, 'batch', None)
        original_ea   = graph.edge_attr
        original_ei   = graph.edge_index
        N = graph.x.shape[0]
        device = graph.x.device
        dtype  = graph.x.dtype

        batch_bc = (original_batch if original_batch is not None
                    else torch.zeros(N, dtype=torch.long, device=device))
        vae_free_bits = float(self.config.get('free_bits', 0.0))
        vae_graph_aware = bool(self.config.get('vae_graph_aware', False))

        use_posterior = self.training and original_y is not None
        if use_posterior:
            z, mu, logvar = self.model.vae_encoder(
                original_y, original_ei, original_ea, batch_bc,
                x=(original_x if vae_graph_aware else None),
            )
            mmd = GNNVariationalEncoder.mmd_loss(z.float())
            kl_clamped, kl_raw = GNNVariationalEncoder.kl_loss(
                mu.float(), logvar.float(), free_bits=vae_free_bits)
            vae_losses = {'mmd': mmd, 'kl': kl_clamped, 'kl_raw': kl_raw}
        else:
            B = int(batch_bc.max().item()) + 1 if original_batch is not None else 1
            z = torch.randn(B, self._vae_latent_dim, device=device, dtype=dtype)
            zero = torch.zeros((), device=device, dtype=torch.float32)
            vae_losses = {'mmd': zero, 'kl': zero, 'kl_raw': zero}

        z_per_node = z[batch_bc]

        aux_loss = torch.zeros((), device=device, dtype=dtype)
        if self.training and use_posterior and original_y is not None:
            B = z.shape[0]
            y_mean = scatter(original_y, batch_bc, dim=0, dim_size=B, reduce='mean')
            y_centered = original_y - y_mean[batch_bc]
            y_std = scatter(y_centered.pow(2), batch_bc, dim=0, dim_size=B, reduce='mean').sqrt()
            aux_target = torch.cat([y_mean, y_std], dim=-1)
            aux_loss = torch.nn.functional.mse_loss(self.model.aux_decoder(z), aux_target)

        return z_per_node, vae_losses, aux_loss

    # ------------------------------------------------------------------
    # Local computation: flat mode
    # ------------------------------------------------------------------

    def _fuse_z(self, x, z_per_node, fuser):
        return fuser(torch.cat([x, z_per_node], dim=-1))

    def run_local_blocks_flat(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        world_edge_attr: Optional[torch.Tensor] = None,
        world_edge_index: Optional[torch.Tensor] = None,
        z_per_node: Optional[torch.Tensor] = None,
    ) -> Tuple:
        graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        if self.use_world_edges and world_edge_attr is not None:
            graph.world_edge_attr  = world_edge_attr
            graph.world_edge_index = world_edge_index

        for i in self.my_block_indices:
            if self.use_vae and z_per_node is not None:
                graph.x = self._fuse_z(graph.x, z_per_node, self.model.z_fusers[str(i)])
            graph = self.model.processer_list[str(i)](graph)

        wea = getattr(graph, 'world_edge_attr',  None)
        wei = getattr(graph, 'world_edge_index', None)
        return graph.x, graph.edge_attr, graph.edge_index, wea, wei

    # ------------------------------------------------------------------
    # Local computation: multiscale mode
    # ------------------------------------------------------------------

    def _extract_level_data(self, graph, level: int) -> dict:
        ld = {
            'ftc': graph[f'fine_to_coarse_{level}'],
            'c_ei': graph[f'coarse_edge_index_{level}'],
            'c_ea': graph[f'coarse_edge_attr_{level}'],
            'n_c': int(graph[f'num_coarse_{level}'].sum()),
            'c_we_idx':  getattr(graph, f'coarse_world_edge_index_{level}', None),
            'c_we_attr': getattr(graph, f'coarse_world_edge_attr_{level}', None),
        }
        if bool(self.config.get('bipartite_unpool', False)):
            up_ei = getattr(graph, f'unpool_edge_index_{level}', None)
            if up_ei is not None:
                ld['up_ei'] = up_ei
                ld['coarse_centroid'] = getattr(graph, f'coarse_centroid_{level}', None)
                ld['fine_pos'] = (graph.pos if level == 0
                                  else getattr(graph, f'coarse_centroid_{level - 1}', None))
        return ld

    def run_local_blocks_multiscale(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        skip_stack: List[dict],
        world_edge_attr: Optional[torch.Tensor],
        world_edge_index: Optional[torch.Tensor],
        z_per_node: Optional[torch.Tensor],
        current_level_idx: int,
        graph,   # local PyG Data object — provides level topology
    ) -> Tuple:
        """Execute the stage's V-cycle ops. Returns updated state."""
        current_graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        if self.use_world_edges and world_edge_attr is not None:
            current_graph.world_edge_attr  = world_edge_attr
            current_graph.world_edge_index = world_edge_index

        bipartite_unpool = bool(self.config.get('bipartite_unpool', False))
        level_idx = current_level_idx  # tracks current coarsening depth

        for op in self._ops_sequence:
            if op[0] == 'block':
                _, kind, level, local_idx = op
                lv, li = str(level), str(local_idx)

                if self.use_vae and z_per_node is not None:
                    if kind == 'pre':
                        fuser = self.model.ms_z_fusers_pre[lv][li]
                    elif kind == 'coarsest':
                        fuser = self.model.ms_z_fusers_coarsest[li]
                    else:  # post
                        fuser = self.model.ms_z_fusers_post[lv][li]
                    current_graph.x = self._fuse_z(current_graph.x, z_per_node, fuser)

                if kind == 'pre':
                    current_graph = self.model.pre_blocks[lv][li](current_graph)
                elif kind == 'coarsest':
                    current_graph = self.model.coarsest_blocks[li](current_graph)
                else:
                    current_graph = self.model.post_blocks[lv][li](current_graph)

            elif op[0] == 'save_pool':
                pool_level = op[1]
                ld = self._extract_level_data(graph, pool_level)
                use_we_here = self.use_world_edges and (
                    pool_level == 0 or self.use_coarse_world_edges)
                # Save skip state BEFORE pooling
                skip_stack.append({
                    'x':         current_graph.x,
                    'edge_attr': current_graph.edge_attr,
                    'edge_index': current_graph.edge_index,
                    'w_attr': (getattr(current_graph, 'world_edge_attr', None)
                               if use_we_here else None),
                    'w_idx':  (getattr(current_graph, 'world_edge_index', None)
                               if use_we_here else None),
                    'z_per_node': z_per_node,
                })
                # Pool
                h_coarse = pool_features(current_graph.x, ld['ftc'], ld['n_c'])
                e_coarse = self.model.coarse_eb_encoders[str(pool_level)](ld['c_ea'])
                current_graph = Data(x=h_coarse, edge_attr=e_coarse, edge_index=ld['c_ei'])
                if self.use_coarse_world_edges:
                    c_we_idx = ld.get('c_we_idx')
                    if c_we_idx is not None and c_we_idx.shape[1] > 0:
                        current_graph.world_edge_attr  = ld['c_we_attr']
                        current_graph.world_edge_index = c_we_idx
                # Update z_per_node for coarser level via scatter_mean
                if self.use_vae and z_per_node is not None:
                    z_per_node = scatter(z_per_node, ld['ftc'], dim=0,
                                         dim_size=ld['n_c'], reduce='mean')
                level_idx += 1

            elif op[0] == 'unpool':
                unpool_level = op[1]
                ld = self._extract_level_data(graph, unpool_level)
                skip = skip_stack[-1]

                up_ei = ld.get('up_ei')
                if (bipartite_unpool and hasattr(self.model, 'unpool_blocks')
                        and up_ei is not None
                        and ld.get('coarse_centroid') is not None
                        and ld.get('fine_pos') is not None):
                    rel_pos = ld['fine_pos'][up_ei[1]] - ld['coarse_centroid'][up_ei[0]]
                    h_up = self.model.unpool_blocks[str(unpool_level)](
                        h_coarse=current_graph.x,
                        h_fine_skip=skip['x'],
                        unpool_edge_index=up_ei,
                        rel_pos=rel_pos,
                    )
                else:
                    h_up = unpool_features(current_graph.x, ld['ftc'])

                h_merged = self.model.skip_projs[str(unpool_level)](
                    torch.cat([skip['x'], h_up], dim=-1))
                current_graph = Data(x=h_merged,
                                     edge_attr=skip['edge_attr'],
                                     edge_index=skip['edge_index'])
                use_we_here = self.use_world_edges and (
                    unpool_level == 0 or self.use_coarse_world_edges)
                if use_we_here and skip.get('w_attr') is not None:
                    current_graph.world_edge_attr  = skip['w_attr']
                    current_graph.world_edge_index = skip['w_idx']

                # Restore z_per_node from skip state (fine-level z)
                if self.use_vae:
                    z_per_node = skip.get('z_per_node')

                skip_stack.pop()
                level_idx -= 1

        wea = getattr(current_graph, 'world_edge_attr',  None)
        wei = getattr(current_graph, 'world_edge_index', None)
        return (current_graph.x, current_graph.edge_attr, current_graph.edge_index,
                skip_stack, wea, wei, z_per_node, level_idx)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, x: torch.Tensor, edge_attr: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
        if not self.is_last:
            raise RuntimeError("decode() called on non-last stage")
        graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        return self.model.decoder(graph)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_stage(
    config: dict,
    stage_idx: int,
    num_stages: int,
    assignment: Sequence[Sequence[int]],
    device: torch.device,
) -> ModelSplitStage:
    return ModelSplitStage(config, stage_idx, num_stages, assignment, device)
