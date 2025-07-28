import torch

try:
    import xformers.ops as xops

    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False
    xops = None
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
        op=None,  # xops.AttentionOp | None
        layer_norm: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.global_rank = 0
        self.world_size = 1

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

        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = None

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def init_ddp_connection(self, global_rank, world_size):
        """Initialize distributed data parallel connection."""
        self.global_rank = global_rank
        self.world_size = world_size

        # Split heads
        self.local_heads = set(
            range(
                global_rank * self.num_heads // world_size,
                (global_rank + 1) * self.num_heads // world_size,
            )
        )

        # Buffers for non-local heads
        self.key_buffer = {}
        self.value_buffer = {}

        # Initialize buffers for each non-local rank
        for rank in range(world_size):
            if rank != global_rank:
                self.key_buffer[rank] = None
                self.value_buffer[rank] = None

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False
    ) -> tuple[Tensor, None]:
        # For single node, no sharding needed
        # For distributed, would shard by batch dimension
        if self.world_size > 1 and torch.distributed.is_initialized():
            batch_size = query.size(0)
            shard_size = batch_size // self.world_size
            start_idx = self.global_rank * shard_size
            end_idx = (
                start_idx + shard_size
                if self.global_rank < self.world_size - 1
                else batch_size
            )

            query = query[start_idx:end_idx]
            key = key[start_idx:end_idx]
            value = value[start_idx:end_idx]

        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_heads)

        # Transfer non-local keys, values to buffers
        if hasattr(self, "key_buffer") and hasattr(self, "value_buffer"):
            # Gather keys and values from all ranks
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            # Create buffers for all-to-all communication
            all_keys = [torch.zeros_like(k) for _ in range(world_size)]
            all_values = [torch.zeros_like(v) for _ in range(world_size)]

            # All-gather keys and values
            torch.distributed.all_gather(all_keys, k.contiguous())
            torch.distributed.all_gather(all_values, v.contiguous())

            # Store non-local keys and values in buffers
            for i in range(world_size):
                if i != rank:
                    # Extract heads for this rank
                    head_start = i * self.num_heads // world_size
                    head_end = (i + 1) * self.num_heads // world_size

                    # Store in buffers for use during attention
                    self.key_buffer[i] = all_keys[i][
                        :, :, head_start:head_end, :
                    ].contiguous()
                    self.value_buffer[i] = all_values[i][
                        :, :, head_start:head_end, :
                    ].contiguous()

        # Sync buffers between GPUs
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

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

    def _reset_parameters(self):
        """Initialize parameters using Xavier initialization."""
        # Initialize linear projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
