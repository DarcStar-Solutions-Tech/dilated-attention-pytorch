#!/usr/bin/env python3
"""
Practical integration of Liquid CfC Router with Ring Dilated Attention.
Shows how to use the actual implementations from the codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

# Import actual implementations from the codebase
try:
    from dilated_attention_pytorch import (
        RingDilatedAttentionV2Collective as RingDilatedAttention,
        create_multihead_dilated_attention,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import Ring Attention. Using placeholders.")
    IMPORTS_AVAILABLE = False


class LiquidCfCExpertRouter(nn.Module):
    """
    Lightweight Liquid CfC router optimized for Ring Attention experts.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        router_dim: int = 256,  # Smaller router state
        num_experts: int = 64,
        top_k: int = 2,
        temporal_weight: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.router_dim = router_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temporal_weight = temporal_weight

        # Input projection (compress to router dimension)
        self.input_proj = nn.Linear(hidden_dim, router_dim)

        # Liquid CfC parameters
        self.tau = nn.Parameter(torch.ones(router_dim))
        self.A = nn.Parameter(torch.randn(router_dim, router_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(router_dim, router_dim) * 0.1)

        # Expert selection
        self.expert_gate = nn.Linear(router_dim, num_experts)

        # Initialize state
        self.register_buffer("hidden_state", None)
        self.register_buffer("prev_selection", None)

    def reset_state(self, batch_size: int, device: torch.device):
        """Reset router state for new sequence."""
        self.hidden_state = torch.zeros(batch_size, self.router_dim, device=device)
        self.prev_selection = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts using Liquid dynamics.

        Args:
            x: Input tensor (batch_size, hidden_dim)

        Returns:
            expert_indices: (batch_size, top_k)
            expert_weights: (batch_size, top_k)
        """
        batch_size = x.size(0)
        device = x.device

        # Initialize state if needed
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.reset_state(batch_size, device)

        # Project input
        x_proj = self.input_proj(x)

        # Liquid CfC dynamics
        dhdt = -self.hidden_state / self.tau + torch.tanh(
            torch.matmul(self.hidden_state, self.A) + torch.matmul(x_proj, self.B)
        )
        self.hidden_state = self.hidden_state + 0.1 * dhdt  # dt = 0.1

        # Expert selection
        expert_logits = self.expert_gate(self.hidden_state)

        # Temporal coherence
        if self.prev_selection is not None:
            # Bias towards previous selection
            prev_bias = torch.zeros_like(expert_logits)
            prev_bias.scatter_(1, self.prev_selection, self.temporal_weight)
            expert_logits = expert_logits + prev_bias

        # Top-k selection
        top_k_values, top_k_indices = torch.topk(expert_logits, self.top_k)
        top_k_weights = F.softmax(top_k_values, dim=-1)

        # Update previous selection
        self.prev_selection = top_k_indices

        return top_k_indices, top_k_weights


class RingAttentionExpertModule(nn.Module):
    """
    Expert module using actual Ring Dilated Attention from the codebase.
    """

    def __init__(
        self,
        expert_id: int,
        embed_dim: int = 4096,
        num_heads: int = 32,
        segment_lengths: List[int] = [8192, 16384, 32768],
        dilation_rates: List[int] = [1, 2, 4],
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_ring_attention: bool = True,
    ):
        super().__init__()
        self.expert_id = expert_id
        self.embed_dim = embed_dim

        if ffn_dim is None:
            ffn_dim = embed_dim * 4

        if use_ring_attention and IMPORTS_AVAILABLE:
            # Use actual Ring Attention implementation
            self.attention = create_multihead_dilated_attention(
                "ring",  # Use ring implementation
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=dropout,
                batch_first=True,
            )
        else:
            # Fallback to standard attention
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        # Expert-specific FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Expert specialization
        self.expert_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass through expert."""
        # Add expert specialization
        x = x + self.expert_embed

        # Self-attention with residual
        if hasattr(self.attention, "forward"):
            attn_out, _ = self.attention(
                x, x, x, attn_mask=attn_mask, is_causal=is_causal
            )
        else:
            attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class LiquidRingMoELayer(nn.Module):
    """
    Complete MoE layer combining Liquid CfC routing with Ring Attention experts.
    """

    def __init__(
        self,
        embed_dim: int = 4096,
        num_experts: int = 64,
        expert_capacity_factor: float = 1.25,
        top_k: int = 2,
        segment_lengths: List[int] = [8192, 16384, 32768],
        dilation_rates: List[int] = [1, 2, 4],
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        router_dim: int = 256,
        temporal_weight: float = 0.1,
        use_ring_attention: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor

        # Liquid CfC router
        self.router = LiquidCfCExpertRouter(
            hidden_dim=embed_dim,
            router_dim=router_dim,
            num_experts=num_experts,
            top_k=top_k,
            temporal_weight=temporal_weight,
        )

        # Ring Attention experts
        self.experts = nn.ModuleList(
            [
                RingAttentionExpertModule(
                    expert_id=i,
                    embed_dim=embed_dim,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    use_ring_attention=use_ring_attention,
                )
                for i in range(num_experts)
            ]
        )

        # Load balancing
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        self.load_balance_loss = 0.0

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_expert_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through Liquid-routed Ring MoE.

        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            attn_mask: Optional attention mask
            is_causal: Whether to use causal masking
            return_expert_info: Whether to return routing information

        Returns:
            Output tensor and optionally expert routing info
        """
        batch_size, seq_len, embed_dim = x.shape

        # Flatten for routing
        x_flat = x.view(-1, embed_dim)

        # Get routing decisions
        expert_indices, expert_weights = self.router(x_flat)

        # Reshape routing
        expert_indices = expert_indices.view(batch_size, seq_len, self.top_k)
        expert_weights = expert_weights.view(batch_size, seq_len, self.top_k)

        # Initialize output
        output = torch.zeros_like(x)

        # Process each expert
        expert_batch_sizes = torch.zeros(
            self.num_experts, dtype=torch.long, device=x.device
        )

        for expert_id in range(self.num_experts):
            # Find all positions assigned to this expert
            expert_mask = (expert_indices == expert_id).any(
                dim=-1
            )  # (batch_size, seq_len)

            if not expert_mask.any():
                continue

            # Get batch indices and sequence positions
            batch_indices, seq_indices = expert_mask.nonzero(as_tuple=True)

            if len(batch_indices) == 0:
                continue

            # Capacity dropping (optional)
            expert_capacity = int(
                self.expert_capacity_factor * len(batch_indices) / self.num_experts
            )
            if len(batch_indices) > expert_capacity:
                # Random dropping for load balancing
                keep_indices = torch.randperm(len(batch_indices))[:expert_capacity]
                batch_indices = batch_indices[keep_indices]
                seq_indices = seq_indices[keep_indices]

            # Gather tokens for this expert
            expert_tokens = x[batch_indices, seq_indices]  # (num_tokens, embed_dim)

            # Process through expert (need to add batch dimension back)
            if len(expert_tokens) > 0:
                # Group by batch for processing
                unique_batches = torch.unique(batch_indices)

                for batch_id in unique_batches:
                    batch_mask = batch_indices == batch_id
                    batch_seq_indices = seq_indices[batch_mask]
                    batch_tokens = expert_tokens[batch_mask]

                    # Process this batch's tokens
                    batch_tokens = batch_tokens.unsqueeze(
                        0
                    )  # (1, num_tokens, embed_dim)
                    expert_output = self.experts[expert_id](
                        batch_tokens, attn_mask=attn_mask, is_causal=is_causal
                    )
                    expert_output = expert_output.squeeze(0)

                    # Get weights and accumulate
                    for i, (seq_idx, token_out) in enumerate(
                        zip(batch_seq_indices, expert_output)
                    ):
                        # Find which k position this expert is for this token
                        k_positions = (
                            expert_indices[batch_id, seq_idx] == expert_id
                        ).nonzero(as_tuple=True)[0]
                        if len(k_positions) > 0:
                            weight = expert_weights[batch_id, seq_idx, k_positions[0]]
                            output[batch_id, seq_idx] += weight * token_out

            # Update counts for load balancing
            expert_batch_sizes[expert_id] = len(batch_indices)

        # Compute load balancing loss
        if self.training:
            target_count = batch_size * seq_len * self.top_k / self.num_experts
            self.load_balance_loss = torch.var(expert_batch_sizes.float()) / (
                target_count**2
            )

        if return_expert_info:
            info = {
                "expert_indices": expert_indices,
                "expert_weights": expert_weights,
                "expert_batch_sizes": expert_batch_sizes,
                "load_balance_loss": self.load_balance_loss,
            }
            return output, info

        return output


def create_liquid_ring_moe_model(
    num_layers: int = 24,
    embed_dim: int = 4096,
    num_heads: int = 32,
    num_experts: int = 64,
    top_k: int = 2,
    segment_lengths: List[int] = [8192, 16384, 32768],
    dilation_rates: List[int] = [1, 2, 4],
    vocab_size: int = 50000,
    max_seq_len: int = 1_000_000,
    use_ring_attention: bool = True,
) -> nn.Module:
    """
    Create a complete transformer model with Liquid Ring MoE layers.
    """

    class LiquidRingTransformer(nn.Module):
        def __init__(self):
            super().__init__()

            # Token embedding
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

            # Transformer layers
            self.layers = nn.ModuleList(
                [
                    LiquidRingMoELayer(
                        embed_dim=embed_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        use_ring_attention=use_ring_attention,
                    )
                    for _ in range(num_layers)
                ]
            )

            # Output projection
            self.ln_f = nn.LayerNorm(embed_dim)
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        def forward(
            self,
            input_ids: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            return_expert_info: bool = False,
        ):
            seq_len = input_ids.size(1)

            # Embeddings
            x = self.token_embed(input_ids)
            x = x + self.pos_embed[:, :seq_len, :]

            # Process through layers
            all_expert_info = []
            for layer in self.layers:
                if return_expert_info:
                    x, info = layer(
                        x, attn_mask=attn_mask, is_causal=True, return_expert_info=True
                    )
                    all_expert_info.append(info)
                else:
                    x = layer(x, attn_mask=attn_mask, is_causal=True)

            # Output
            x = self.ln_f(x)
            logits = self.lm_head(x)

            if return_expert_info:
                return logits, all_expert_info
            return logits

    return LiquidRingTransformer()


def demonstrate_liquid_ring_integration():
    """Demonstrate the practical integration."""
    print("=== Liquid CfC + Ring Attention Integration Demo ===\n")

    # Small configuration for demo
    batch_size = 2
    seq_len = 512
    embed_dim = 768
    num_experts = 16

    print("Creating Liquid Ring MoE layer...")
    moe_layer = LiquidRingMoELayer(
        embed_dim=embed_dim,
        num_experts=num_experts,
        top_k=2,
        segment_lengths=[128, 256, 512],
        dilation_rates=[1, 2, 4],
        use_ring_attention=IMPORTS_AVAILABLE,
    )

    print("Configuration:")
    print(f"  - Embed dimension: {embed_dim}")
    print(f"  - Number of experts: {num_experts}")
    print("  - Experts per token: 2")
    print(f"  - Using Ring Attention: {IMPORTS_AVAILABLE}")

    # Test input
    x = torch.randn(batch_size, seq_len, embed_dim)

    print(f"\nProcessing input: {x.shape}")

    # Forward pass
    output, info = moe_layer(x, is_causal=True, return_expert_info=True)

    print(f"Output shape: {output.shape}")
    print("\nRouting statistics:")
    print(f"  - Expert indices shape: {info['expert_indices'].shape}")
    print(f"  - Expert weights shape: {info['expert_weights'].shape}")

    # Analyze load distribution
    batch_sizes = info["expert_batch_sizes"].cpu().numpy()
    print("\nExpert load distribution:")
    print(f"  - Mean tokens/expert: {batch_sizes.mean():.1f}")
    print(f"  - Std deviation: {batch_sizes.std():.1f}")
    print(f"  - Min/Max: {batch_sizes.min()}/{batch_sizes.max()}")
    print(f"  - Load balance loss: {info['load_balance_loss']:.6f}")

    # Test temporal coherence
    print("\nTesting temporal coherence...")
    router = moe_layer.router
    prev_indices = None
    switches = 0

    for t in range(10):
        x_t = torch.randn(1, embed_dim)
        indices, _ = router(x_t)

        if prev_indices is not None and not torch.equal(indices, prev_indices):
            switches += 1
        prev_indices = indices

    print(f"Expert switches in 10 steps: {switches} ({switches / 10 * 100:.0f}%)")

    print("\n✅ Integration test complete!")

    # Usage example
    print("\n" + "=" * 70)
    print("USAGE EXAMPLE")
    print("=" * 70)
    print("""
    # Create full model
    model = create_liquid_ring_moe_model(
        num_layers=48,
        embed_dim=8192,
        num_experts=256,
        top_k=2,
        segment_lengths=[65536, 131072, 262144],  # Up to 256K chunks
        max_seq_len=1_000_000_000,  # 1B tokens!
        use_ring_attention=True
    )
    
    # Distributed setup (pseudo-code)
    model = DistributedDataParallel(model)
    
    # Training
    for batch in dataloader:
        logits, expert_info = model(batch['input_ids'], return_expert_info=True)
        
        # Language modeling loss
        lm_loss = F.cross_entropy(logits.flatten(0, 1), batch['labels'].flatten())
        
        # Add load balancing losses
        aux_loss = sum(info['load_balance_loss'] for layer_info in expert_info 
                      for info in layer_info)
        
        total_loss = lm_loss + 0.01 * aux_loss
        total_loss.backward()
    """)


if __name__ == "__main__":
    demonstrate_liquid_ring_integration()
