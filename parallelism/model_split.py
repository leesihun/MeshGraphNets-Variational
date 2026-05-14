"""ModelSplitStage — one pipeline stage of a sliced MeshGraphNets.

Each rank owns:
  - Stage 0: Encoder + assigned GnBlocks
  - Middle stages: assigned GnBlocks only
  - Last stage: assigned GnBlocks + Decoder

Cross-rank dance (send/recv + autograd bridging) lives in `parallelism/launcher.py`.
This file exposes the local computation building blocks and the autograd
Functions that bridge gradient flow across ranks.

MVP scope:
  - Flat processor only (use_multiscale=False)
  - No VAE (use_vae=False)
  - No world edges (use_world_edges=False)
Unsupported configs raise NotImplementedError at construction time.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch_geometric.data import Data

from general_modules.edge_features import EDGE_FEATURE_DIM
from model.encoder_decoder import Decoder, Encoder, GnBlock


# ---------------------------------------------------------------------------
# Wire-level send/recv with shape header
# ---------------------------------------------------------------------------
# IMPORTANT: NCCL ignores `tag`. send/recv pairs are matched strictly by order,
# so within a forward (or backward) pass, both ranks MUST issue their sends
# and recvs in the exact same sequence.

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
# Autograd bridges
# ---------------------------------------------------------------------------
# A pair of `dist.send/recv` calls is bundled into a single autograd Function
# so the send/recv order is deterministic on both ranks (no race on NCCL).
#
# Forward flow:  x, edge_attr (grad-bearing) + edge_index (no grad)
# Backward flow: grad_x, grad_edge_attr (in the same order as forward)


class SendActivations(torch.autograd.Function):
    """Send (x, edge_attr) downstream; receive their grads in backward.

    `edge_index` is sent inside this Function too but is non-differentiable,
    so it has no entry in the backward grad list.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, edge_attr: torch.Tensor,
                edge_index: torch.Tensor, dst: int) -> torch.Tensor:
        ctx.dst = dst
        ctx.device = x.device
        ctx.x_shape = x.shape
        ctx.ea_shape = edge_attr.shape
        # Strict send order: x, edge_attr, edge_index
        _send_tensor(x.contiguous(), dst=dst)
        _send_tensor(edge_attr.contiguous(), dst=dst)
        _send_tensor(edge_index.contiguous(), dst=dst)
        # Sentinel scalar to anchor backward
        return torch.zeros((), device=x.device, dtype=x.dtype, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        # Strict recv order matches the OTHER side's send order in its backward:
        # grad_x first, then grad_edge_attr. edge_index has no grad.
        grad_x = _recv_tensor(src=ctx.dst, device=ctx.device)
        grad_ea = _recv_tensor(src=ctx.dst, device=ctx.device)
        return grad_x, grad_ea, None, None


class RecvActivations(torch.autograd.Function):
    """Receive (x, edge_attr, edge_index) from upstream; send their grads in backward."""

    @staticmethod
    def forward(ctx, src: int, device: torch.device, anchor: torch.Tensor):
        ctx.src = src
        ctx.device = device
        # Strict recv order: x, edge_attr, edge_index (matches sender)
        x = _recv_tensor(src=src, device=device)
        ea = _recv_tensor(src=src, device=device)
        ei = _recv_tensor(src=src, device=device)
        # edge_index is int64; tell autograd it cannot carry grad.
        ctx.mark_non_differentiable(ei)
        return x, ea, ei

    @staticmethod
    def backward(ctx, grad_x, grad_ea, grad_ei):
        # grad_ei is None (ei was marked non-differentiable).
        # Strict send order matches the OTHER side's recv order in its backward:
        # grad_x first, then grad_edge_attr.
        _send_tensor(grad_x.contiguous(), dst=ctx.src)
        _send_tensor(grad_ea.contiguous(), dst=ctx.src)
        return None, None, None


def send_to_next(x: torch.Tensor, edge_attr: torch.Tensor,
                 edge_index: torch.Tensor, dst: int) -> torch.Tensor:
    """Send (x, edge_attr, edge_index) to rank `dst`. Returns a 0-dim sentinel
    that participates in the autograd graph; call `.sum().backward()` on it
    (or include it in a loss expression) to trigger gradient receipt."""
    return SendActivations.apply(x, edge_attr, edge_index, dst)


def recv_from_prev(src: int, device: torch.device):
    """Receive (x, edge_attr, edge_index) from rank `src`. x and edge_attr
    require grad; edge_index does not."""
    anchor = torch.zeros((), device=device, requires_grad=True)
    return RecvActivations.apply(src, device, anchor)


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
        self._validate_config(config)

        self.config = config
        self.stage_idx = int(stage_idx)
        self.num_stages = int(num_stages)
        self.is_first = (self.stage_idx == 0)
        self.is_last = (self.stage_idx == self.num_stages - 1)
        self.device = device

        my_blocks = sorted(assignment[stage_idx])
        self.my_block_indices = my_blocks
        L = int(config['message_passing_num'])
        if my_blocks and (my_blocks[0] < 0 or my_blocks[-1] >= L):
            raise ValueError(f"stage {stage_idx} block indices {my_blocks} out of range for L={L}")

        latent_dim = int(config['latent_dim'])
        edge_input_size = int(config['edge_var'])
        if edge_input_size != EDGE_FEATURE_DIM:
            raise ValueError(f"edge_var must be {EDGE_FEATURE_DIM}, got {edge_input_size}")

        base_input_size = int(config['input_var'])
        base_input_size += int(config.get('positional_features', 0))
        if config.get('use_node_types', False) and int(config.get('num_node_types', 0)) > 0:
            node_input_size = base_input_size + int(config['num_node_types'])
        else:
            node_input_size = base_input_size

        self.model = _StageInner(
            is_first=self.is_first,
            is_last=self.is_last,
            my_blocks=my_blocks,
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            node_output_size=int(config['output_var']),
            latent_dim=latent_dim,
            config=config,
        )

        self.to(device)

        # Decoder last-layer scale (matches MeshGraphNets.__init__)
        num_timesteps = config.get('num_timesteps', None)
        if (num_timesteps is None or num_timesteps > 1) and self.is_last:
            with torch.no_grad():
                self.model.decoder.decode_module[-1].weight.mul_(0.01)

    @staticmethod
    def _validate_config(config: dict) -> None:
        if config.get('use_multiscale', False):
            raise NotImplementedError(
                "parallel_mode=model_split: use_multiscale=True is not supported yet."
            )
        if config.get('use_vae', False):
            raise NotImplementedError(
                "parallel_mode=model_split: use_vae=True is not supported yet."
            )
        if config.get('use_world_edges', False):
            raise NotImplementedError(
                "parallel_mode=model_split: use_world_edges=True is not supported yet."
            )

    # -- Local computation ---------------------------------------------------

    def apply_input_noise(self, graph) -> None:
        """Apply training-time input noise (stage 0 only); mirrors MeshGraphNets.forward."""
        noise_std = self.config.get('std_noise', 0.0)
        if noise_std <= 0:
            return
        output_var = int(self.config['output_var'])
        noise = torch.randn(graph.x.shape[0], output_var, device=graph.x.device, dtype=graph.x.dtype) * noise_std
        noise_padded = torch.zeros_like(graph.x)
        noise_padded[:, :output_var] = noise
        graph.x = graph.x + noise_padded
        noise_gamma = self.config.get('noise_gamma', 0.1)
        noise_std_ratio = self.config.get('noise_std_ratio', None)
        if noise_std_ratio is not None:
            ratio = torch.tensor(noise_std_ratio, device=graph.x.device, dtype=graph.x.dtype)
            graph.y = graph.y - noise_gamma * noise * ratio
        graph.edge_attr = graph.edge_attr + torch.randn_like(graph.edge_attr) * noise_std

    def encode(self, graph) -> Data:
        if not self.is_first:
            raise RuntimeError("encode() called on non-first stage")
        return self.model.encoder(graph)

    def run_local_blocks(self, x: torch.Tensor, edge_attr: torch.Tensor,
                         edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        for i in self.my_block_indices:
            graph = self.model.processer_list[str(i)](graph)
        return graph.x, graph.edge_attr, graph.edge_index

    def decode(self, x: torch.Tensor, edge_attr: torch.Tensor,
               edge_index: torch.Tensor) -> torch.Tensor:
        if not self.is_last:
            raise RuntimeError("decode() called on non-last stage")
        graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        return self.model.decoder(graph)


class _StageInner(nn.Module):
    """Holds encoder/processer_list/decoder for ONE stage.

    Uses ModuleDict for processer_list keyed by stringified block index, so the
    state_dict keys match the full single-GPU model's `model.processer_list.{i}.*`
    namespace and slice/merge cleanly.
    """

    def __init__(
        self,
        *,
        is_first: bool,
        is_last: bool,
        my_blocks: Sequence[int],
        edge_input_size: int,
        node_input_size: int,
        node_output_size: int,
        latent_dim: int,
        config: dict,
    ):
        super().__init__()
        use_world_edges = bool(config.get('use_world_edges', False))

        if is_first:
            self.encoder = Encoder(
                edge_input_size, node_input_size, latent_dim,
                use_world_edges=use_world_edges,
            )
        self.processer_list = nn.ModuleDict()
        for i in my_blocks:
            self.processer_list[str(i)] = GnBlock(config, latent_dim, use_world_edges=use_world_edges)
        if is_last:
            self.decoder = Decoder(latent_dim, node_output_size)


def build_stage(
    config: dict,
    stage_idx: int,
    num_stages: int,
    assignment: Sequence[Sequence[int]],
    device: torch.device,
) -> ModelSplitStage:
    return ModelSplitStage(config, stage_idx, num_stages, assignment, device)
