"""
Dilated Attention PyTorch Implementation

Unofficial PyTorch implementation of DilatedAttention from LongNet, 
including Ring Attention for O(n) memory scaling.
"""

__version__ = "0.1.0"

from .dilated_attention import DilatedAttention
from .multihead_dilated_attention import MultiheadDilatedAttention
from .improved_dilated_attention import ImprovedDilatedAttention
from .distributed_dilated_attention import DistributedDilatedAttention
from .ring_dilated_attention import RingDilatedAttention
from .ring_multihead_dilated_attention import RingMultiheadDilatedAttention
from .transformer import Transformer
from .long_net import LongNet

__all__ = [
    "DilatedAttention",
    "MultiheadDilatedAttention", 
    "ImprovedDilatedAttention",
    "DistributedDilatedAttention",
    "RingDilatedAttention",
    "RingMultiheadDilatedAttention",
    "Transformer",
    "LongNet",
]