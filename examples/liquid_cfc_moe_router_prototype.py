#!/usr/bin/env python3
"""
Prototype implementation of Liquid CfC Router for MoE with Ring Attention.
Based on the CfC (Closed-form Continuous-time) neural network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CfCCell(nn.Module):
    """
    Closed-form Continuous-time Cell.
    Implements the CfC dynamics without numerical ODE solving.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sparsity: float = 0.5,
        backbone_activation: str = "gelu",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Time constant network
        self.ff1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.ff2 = nn.Linear(hidden_size, hidden_size)
        self.time_a = nn.Linear(hidden_size, hidden_size)
        self.time_b = nn.Linear(hidden_size, hidden_size)

        # Input backbone
        self.input_w = nn.Linear(input_size, hidden_size)
        self.input_b = nn.Parameter(torch.zeros(hidden_size))

        # Recurrent backbone
        self.r_w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.r_b = nn.Parameter(torch.zeros(hidden_size))

        # Initialize with sparsity
        if sparsity > 0:
            self._apply_sparsity(sparsity)

        # Activation
        if backbone_activation == "gelu":
            self.backbone_activation = nn.GELU()
        elif backbone_activation == "relu":
            self.backbone_activation = nn.ReLU()
        else:
            self.backbone_activation = nn.Identity()

    def _apply_sparsity(self, sparsity: float):
        """Apply structured sparsity to weight matrices."""
        with torch.no_grad():
            # Make recurrent weights sparse
            mask = torch.rand(self.hidden_size, self.hidden_size) > sparsity
            self.r_w.weight.data *= mask.float()

    def forward(
        self, input: torch.Tensor, hx: torch.Tensor, ts: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of CfC cell.

        Args:
            input: Input tensor (batch, input_size)
            hx: Hidden state (batch, hidden_size)
            ts: Time step (default 1.0)

        Returns:
            new_h: New hidden state
            output: Cell output (same as new_h for basic cell)
        """
        _ = input.size(0)

        # Compute time constants
        x_cat = torch.cat([input, hx], dim=1)
        ff1_out = self.backbone_activation(self.ff1(x_cat))
        ff2_out = self.backbone_activation(self.ff2(ff1_out))

        # Time constant gating
        t_a = self.time_a(ff2_out)
        t_b = self.time_b(ff2_out)
        t_interp = torch.sigmoid(t_a * ts + t_b)

        # Input and recurrent contributions
        input_contribution = self.backbone_activation(
            self.input_w(input) + self.input_b
        )

        recurrent_contribution = self.backbone_activation(self.r_w(hx) + self.r_b)

        # CfC dynamics (closed-form solution)
        new_h = hx * (1 - t_interp) + t_interp * (
            input_contribution + recurrent_contribution
        )

        return new_h, new_h


class LiquidCfCRouter(nn.Module):
    """
    Liquid CfC-based router for MoE with temporal coherence.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 512,
        num_experts: int = 128,
        top_k: int = 2,
        use_load_balancing: bool = True,
        temporal_smoothing: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_load_balancing = use_load_balancing
        self.temporal_smoothing = temporal_smoothing

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # CfC cell for temporal dynamics
        self.cfc_cell = CfCCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            sparsity=0.5,
            backbone_activation="gelu",
        )

        # Expert gating from CfC hidden state
        self.expert_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, num_experts),
        )

        # Load balancing auxiliary loss
        self.load_balance_loss = 0.0

        # State tracking
        self.register_buffer("hidden_state", None)
        self.register_buffer("prev_expert_probs", None)

    def reset_state(self, batch_size: int, device: torch.device):
        """Reset the hidden state."""
        self.hidden_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.prev_expert_probs = None

    def forward(
        self, x: torch.Tensor, return_all_probs: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route input to experts using Liquid CfC dynamics.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            return_all_probs: Whether to return probabilities for all experts

        Returns:
            expert_indices: Selected expert indices (batch_size, seq_len, top_k)
            expert_weights: Weights for selected experts (batch_size, seq_len, top_k)
            all_probs: Probabilities for all experts if requested
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.reset_state(batch_size, device)

        # Process sequence through CfC
        all_expert_indices = []
        all_expert_weights = []
        all_expert_probs = []

        for t in range(seq_len):
            # Project input
            x_t = self.input_projection(x[:, t, :])

            # Update CfC state
            self.hidden_state, _ = self.cfc_cell(x_t, self.hidden_state)

            # Compute expert logits
            expert_logits = self.expert_gate(self.hidden_state)
            expert_probs = F.softmax(expert_logits, dim=-1)

            # Temporal smoothing with previous timestep
            if self.prev_expert_probs is not None and self.temporal_smoothing > 0:
                expert_probs = (
                    1 - self.temporal_smoothing
                ) * expert_probs + self.temporal_smoothing * self.prev_expert_probs

            self.prev_expert_probs = expert_probs.detach()

            # Select top-k experts
            top_k_probs, top_k_indices = torch.topk(expert_probs, self.top_k, dim=-1)

            # Renormalize top-k probabilities
            top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            all_expert_indices.append(top_k_indices)
            all_expert_weights.append(top_k_weights)
            all_expert_probs.append(expert_probs)

        # Stack results
        expert_indices = torch.stack(all_expert_indices, dim=1)
        expert_weights = torch.stack(all_expert_weights, dim=1)

        # Compute load balancing loss if enabled
        if self.use_load_balancing and self.training:
            self._compute_load_balance_loss(expert_indices, expert_weights)

        if return_all_probs:
            all_probs = torch.stack(all_expert_probs, dim=1)
            return expert_indices, expert_weights, all_probs
        else:
            return expert_indices, expert_weights, None

    def _compute_load_balance_loss(self, indices: torch.Tensor, weights: torch.Tensor):
        """Compute auxiliary loss for load balancing."""
        batch_size, seq_len, _ = indices.shape

        # Count expert usage
        expert_counts = torch.zeros(self.num_experts, device=indices.device)

        # Accumulate weighted expert usage
        for i in range(self.num_experts):
            mask = (indices == i).float()
            expert_counts[i] = (mask * weights).sum()

        # Target uniform distribution
        target_count = batch_size * seq_len * self.top_k / self.num_experts

        # Compute loss (encourage uniform distribution)
        self.load_balance_loss = torch.mean((expert_counts - target_count) ** 2) / (
            target_count**2
        )


class LiquidMoELayer(nn.Module):
    """
    Complete MoE layer with Liquid CfC routing.
    This is a simplified version for demonstration.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        num_experts: int = 8,
        top_k: int = 2,
        expert_capacity_factor: float = 1.25,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor

        # Liquid CfC router
        self.router = LiquidCfCRouter(
            input_dim=input_dim,
            hidden_dim=512,  # Smaller hidden state
            num_experts=num_experts,
            top_k=top_k,
        )

        # Simple experts (in practice, these would be RingAttentionExperts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, input_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor (batch_size, seq_len, input_dim)

        Returns:
            Output tensor same shape as input
        """
        batch_size, seq_len, input_dim = x.shape

        # Get routing decisions
        expert_indices, expert_weights, _ = self.router(x)

        # Flatten for processing
        x_flat = x.view(-1, input_dim)
        output_flat = torch.zeros_like(x_flat)

        # Process each expert
        for i in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = expert_indices == i
            if not expert_mask.any():
                continue

            # Get indices and weights
            expert_token_indices = expert_mask.nonzero(as_tuple=True)
            batch_idx = expert_token_indices[0]
            seq_idx = expert_token_indices[1]
            k_idx = expert_token_indices[2]

            # Linear indices for flat tensors
            flat_indices = batch_idx * seq_len + seq_idx

            # Get tokens for this expert
            expert_tokens = x_flat[flat_indices]

            # Process through expert
            expert_output = self.experts[i](expert_tokens)

            # Get weights for accumulation
            weights = expert_weights[batch_idx, seq_idx, k_idx].unsqueeze(-1)

            # Accumulate weighted outputs
            output_flat.index_add_(0, flat_indices, expert_output * weights)

        # Reshape to original
        output = output_flat.view(batch_size, seq_len, input_dim)

        return output


# Example usage and testing
def test_liquid_router():
    """Test the Liquid CfC router implementation."""
    print("Testing Liquid CfC Router for MoE...")

    # Configuration
    batch_size = 2
    seq_len = 128
    input_dim = 4096
    num_experts = 8

    # Create router
    router = LiquidCfCRouter(
        input_dim=input_dim,
        hidden_dim=256,
        num_experts=num_experts,
        top_k=2,
    )

    # Test input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    expert_indices, expert_weights, all_probs = router(x, return_all_probs=True)

    print(f"Input shape: {x.shape}")
    print(f"Expert indices shape: {expert_indices.shape}")
    print(f"Expert weights shape: {expert_weights.shape}")
    print(f"All probabilities shape: {all_probs.shape}")

    # Check temporal coherence
    print("\nTemporal coherence check:")
    # Count expert switches between timesteps
    switches = (expert_indices[:, 1:] != expert_indices[:, :-1]).float().mean()
    print(f"Average expert switches per timestep: {switches:.3f}")

    # Check load balancing
    print("\nLoad balancing check:")
    expert_usage = torch.zeros(num_experts)
    for i in range(num_experts):
        expert_usage[i] = (expert_indices == i).float().mean()
    print(f"Expert usage: {expert_usage}")
    print(f"Usage std dev: {expert_usage.std():.3f}")

    # Test full MoE layer
    print("\n\nTesting full MoE layer...")
    moe_layer = LiquidMoELayer(
        input_dim=input_dim,
        num_experts=num_experts,
        top_k=2,
    )

    output = moe_layer(x)
    print(f"MoE output shape: {output.shape}")

    # Test gradient flow
    loss = output.mean()
    loss.backward()
    print("Gradient flow successful!")

    # Check router's load balance loss
    print(f"Load balance loss: {router.load_balance_loss:.4f}")


if __name__ == "__main__":
    test_liquid_router()

    print("\n" + "=" * 60)
    print("Integration with Ring Attention:")
    print("=" * 60)
    print("""
    To integrate with Ring Attention, replace the simple expert FFN with:
    
    ```python
    class RingAttentionExpert(nn.Module):
        def __init__(self, expert_id, ring_size=8):
            super().__init__()
            self.attention = RingDilatedAttention(
                ring_size=ring_size,
                segment_lengths=[8192, 16384, 32768],
                use_original=True  # For massive context
            )
            self.ffn = ExpertFFN(hidden_dim=4096)
            
        def forward(self, x):
            # Process with Ring Attention
            attn_out = self.attention(x)
            return self.ffn(attn_out)
    ```
    
    This gives each expert billion-token context capability!
    """)
