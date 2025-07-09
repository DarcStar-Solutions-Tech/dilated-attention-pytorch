import torch
import xformers.ops as xops
from einops import rearrange

# from pytorch_lightning import LightningModule  # Optional dependency
from torch import Tensor, nn

from .dilated_attention import DilatedAttention


class DistributedMultiheadDilatedAttention(
    nn.Module
):  # Can inherit from LightningModule if available
    def __init__(
        self,
        embed_dim,
        num_heads,
        dilation_rates,
        segment_lengths,
        dropout: float = 0.0,
        op: xops.AttentionOp | None = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Other modules
        self.dilated_attentions = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            attention_dropout=dropout,
            op=op,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def init_ddp_connection(self, global_rank, world_size):
        # Split heads
        self.local_heads = set(
            range(
                global_rank * self.num_heads // world_size,
                (global_rank + 1) * self.num_heads // world_size,
            )
        )

        # Buffers for non-local heads
        self.key_buffer = {...}
        self.value_buffer = {...}

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False
    ) -> tuple[Tensor, None]:
        # Shard input
        query = query[self.global_rank * len(query) // self.world_size : ...]
        key = key[self.global_rank * len(key) // self.world_size : ...]
        value = value[self.global_rank * len(value) // self.world_size : ...]

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_heads)

        # Transfer non-local keys, values to buffers
        # TODO: Implement distributed buffer transfer
        # self.key_buffer[dst] = ...
        # self.value_buffer[dst] = ...

        # Sync buffers between GPUs
        torch.distributed.barrier()
        ...

        # Apply attention, then fold 'h' attention heads back into 'd'.
        x = self.dilated_attentions(q, k, v, is_causal=is_causal)
        x = rearrange(x, "b n h d -> b n (h d)")

        # NOTE: This is different from 'nn.MultiheadAttention'! The LongNet paper
        # follows the MAGNETO architecture, which applies an extra layer norm
        # before the linear output projection.  The cross-attention layer in the
        # MAGNETO decoder does not include this layer norm, so users have the option
        # to disable it (layer_norm=False).
        if self.layer_norm:
            assert self.norm is not None
            x = self.norm(x)
        # Linear projection on attention outputs.
        x = self.out_proj(x)

        return x, None
