import logging
import os
from functools import partial
from math import ceil
from timeit import Timer
from typing import Callable, List, NamedTuple

import datetime
import plotly.graph_objects as go
import torch
import xformers.ops as xops

from dilated_attention_pytorch.dilated_attention import MultiheadDilatedAttention

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Benchmarking parameters')

# Add the arguments
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for benchmarking')
parser.add_argument('--total_tokens', type=int, default=2**26, help='Total tokens for benchmarking, default is 64M(2**26) tokens')
parser.add_argument('--num_heads', type=int, default=2, help='Number of heads for benchmarking')
parser.add_argument('--embed_dim', type=int, default=16, help='Embedding dimension for benchmarking')
parser.add_argument('--vanilla_seq_lengths', nargs='+', type=int, default=[2**i for i in range(13, 18)], help='Sequence lengths for vanilla attention')
parser.add_argument('--segment_lengths', nargs='+', type=int, default=[8192, 16384, 32768, 65536], help='Segment lengths for dilated attention')
parser.add_argument('--dilated_seq_lengths', nargs='+', type=int, default=[2**i for i in range(13, 27)], help='Sequence lengths for dilated attention, default sequence lengths are [2**13,2**27) ')

# Parse the arguments
args = parser.parse_args()

# Now you can use the arguments as:
# args.batch_size, args.total_tokens, etc.


# Generic benchmarking parameters
BATCH_SIZE = args.batch_size
TOTAL_TOKENS = args.total_tokens  # 64M
NUM_HEADS = args.num_heads
EMBED_DIM = args.embed_dim
# Vanilla attention only
VANILLA_SEQ_LENGTHS = args.vanilla_seq_lengths  # 8k - 128k

# Dilated attention only
SEGMENT_LENGTHS = args.segment_lengths  # 8k - 64k
DILATED_SEQ_LENGTHS = args.dilated_seq_lengths  # 8k - 64M


class BenchmarkResult(NamedTuple):
    mean: float
    std: float

    def __repr__(self):
        return f"BenchmarkResult(mean: {self.mean:.3e}, std: {self.std:.3e})"

    def __str__(self):
        return f"({self.mean:.3e} \u00B1 {self.std:.3e}) s"


def benchmark(
        fn: Callable,
        *args,
        min_total_seconds: float = 1.0,
        min_iterations: int = 2,
        **kwargs,
) -> BenchmarkResult:
    # Benchmark the runtime of a function and dynamically determine the number of
    # iterations to run.  Continue running the function until *total* runtime
    # exceeds 'min_total_seconds' and 'min_iterations'.
    if min_iterations < 2:
        raise ValueError("min_iterations must be >= 2")

    timer = Timer(
        "fn(*args, **kwargs)",
        globals={"fn": fn, "args": args, "kwargs": kwargs},
    )
    # Run the function once to warm up
    _ = timer.repeat(number=1, repeat=1)

    times: List[float] = []
    total_time = 0.0
    num_iterations = min_iterations or 1

    while total_time < min_total_seconds:
        _times = timer.repeat(number=1, repeat=num_iterations)
        times.extend(_times)
        _total_time = sum(_times)
        total_time += _total_time

        # Estimate how many more iterations we need to run to get to 1 second
        avg_time = _total_time / num_iterations
        num_iterations = ceil((min_total_seconds - total_time) / avg_time)

    times_tensor = torch.as_tensor(times)
    return BenchmarkResult(
        mean=times_tensor.mean().item(),
        std=times_tensor.std().item(),
    )


def get_dilated_attention_for_seq_length(seq_length: int) -> MultiheadDilatedAttention:
    """This is roughly how benchmarking was described in the paper, except that they
    were testing in a distributed (multi-GPU) setting.  We use a base segment
    length of 8192, and include larger segment lengths if possible.  I believe
    this is the equivalent benchmark for 1 GPU.

    Reference:
        https://arxiv.org/pdf/2307.02486.pdf, Section 3.1
    """
    segment_lengths: List[int] = []
    dilation_rates: List[int] = []

    for segment_length in SEGMENT_LENGTHS:
        # We can't use segment lengths larger than the sequence length.
        segment_length = min(segment_length, seq_length)
        exponent = segment_length // SEGMENT_LENGTHS[0] - 1
        dilation_rate = 2 ** exponent

        segment_lengths.append(segment_length)
        dilation_rates.append(dilation_rate)

    return MultiheadDilatedAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        op=xops.MemoryEfficientAttentionFlashAttentionOp,
        device="cuda",
        dtype=torch.float16
    )


def attention_forward(x: torch.Tensor, attn: Callable):
    with torch.no_grad():
        y = attn(x, x, x, is_causal=False)
        print(y.shape)
    torch.cuda.synchronize()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    token_count = f"{ceil(TOTAL_TOKENS / 2 ** 20)}M"

    logging.info(f"Benchmark vanilla attention against {token_count} tokens...")
    vanilla_results: List[BenchmarkResult] = []
    for seq_length in VANILLA_SEQ_LENGTHS:
        torch.cuda.empty_cache()
        batch_size = TOTAL_TOKENS // seq_length
        x = torch.randn(
            (batch_size, seq_length, NUM_HEADS, EMBED_DIM),
            dtype=torch.float16,
            device="cuda",
        )
        fn = partial(attention_forward, attn=xops.memory_efficient_attention)
        result = benchmark(fn, x)
        vanilla_results.append(result)
        logging.info(f"Sequence length {seq_length}: {result}")

    logging.info(f"Benchmark dilated attention against {token_count} tokens...")
    dilated_results: List[BenchmarkResult] = []
    for seq_length in DILATED_SEQ_LENGTHS:
        torch.cuda.empty_cache()
        batch_size = TOTAL_TOKENS // seq_length
        if batch_size > 0:
            x = torch.randn(
                (batch_size, seq_length, NUM_HEADS, EMBED_DIM),
                dtype=torch.float16,
                device="cuda",
            )
            attn = get_dilated_attention_for_seq_length(seq_length)
            fn = partial(attention_forward, attn=attn)
            result = benchmark(fn, x)
            dilated_results.append(result)
            logging.info(f"Sequence length {seq_length}: {result}")

    # Get current date
    current_date = datetime.date.today()

    logging.info(f"Plotting results for {token_count} tokens...")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=VANILLA_SEQ_LENGTHS,
            y=[r.mean for r in vanilla_results],
            error_y=dict(
                type="data",
                array=[r.std for r in vanilla_results],
                visible=True,
            ),
            name="Vanilla Attention",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=DILATED_SEQ_LENGTHS,
            y=[r.mean for r in dilated_results],
            error_y=dict(
                type="data",
                array=[r.std for r in dilated_results],
                visible=True,
            ),
            name="Dilated Attention",
        ),
    )
    fig.update_layout(
        title=f"Attention Benchmark on {current_date} (Total Tokens = {token_count})",
        xaxis_title="Sequence Length",
        yaxis_title="Runtime (s)",
        xaxis_type="log",
        yaxis_type="log",
    )
    fig.write_image(os.path.join("doc", f"benchmark-{token_count}-tokens-{current_date}.png"))