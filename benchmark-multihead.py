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
from torch import device

from dilated_attention_pytorch.dilated_attention import MultiheadDilatedAttention, DilatedAttention

import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Benchmarking parameters')

# Add the arguments
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for benchmarking')
parser.add_argument('--total_tokens', type=int, default=26,
                    help='Exponent for Total tokens for benchmarking, default is 26 which is 64M(2**26) tokens')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for benchmarking, default is 8')
parser.add_argument('--embed_dim', type=int, default=32, help='Embed dimension for benchmarking, default is 256')
parser.add_argument('--vanilla_seq_lengths', type=int, default=18,
                    help='End of Sequence length range for vanilla attention, default is 18 which is 2**18')
parser.add_argument('--benchmark_vanilla', type=bool, default=True, help='Benchmark vanilla attention, default is True')
parser.add_argument('--segment_lengths', nargs='+', type=int, default=[8192, 16384, 32768, 65536],
                    help='Segment lengths for dilated attention')
parser.add_argument('--dilated_seq_lengths', type=int, default=27,
                    help='End of Sequence lengths for dilated attention, default is 27 which is 2**27 ')
parser.add_argument('--benchmark_dilated', type=bool, default=True, help='Benchmark dilated attention, default is True')
parser.add_argument('--benchmark_multihead', type=bool, default=True,
                    help='Benchmark multihead dilated attention, default is True')

# Parse the arguments
args = parser.parse_args()

# Generic benchmarking parameters
BATCH_SIZE: int = args.batch_size
TOTAL_TOKENS: int = 2 ** args.total_tokens  # 64M
NUM_HEADS: int = args.num_heads
EMBED_DIM: int = args.embed_dim
# Vanilla attention only
VANILLA_SEQ_LENGTHS: List[int] = [2 ** i for i in range(13, args.vanilla_seq_lengths)]  # 8k - 128k

# Dilated attention only
SEGMENT_LENGTHS: List[int] = args.segment_lengths  # 8k - 64k
DILATED_SEQ_LENGTHS: List[int] = [2 ** i for i in args.dilated_seq_lengths]  # 8k - 64M


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


def get_dilated_attention_for_seq_length(seq_length: int) -> DilatedAttention:
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

    return DilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        op=xops.MemoryEfficientAttentionFlashAttentionOp,
    )


def get_multihead_dilated_attention_for_seq_length(seq_length: int) -> MultiheadDilatedAttention:
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
        _ = attn(x, x, x)
    torch.cuda.synchronize()


def plot_results(seq_lengths: List[int], results: List[BenchmarkResult], name: str):
    fig.add_trace(
        go.Scatter(
            x=seq_lengths,
            y=[r.mean for r in results],
            error_y=dict(
                type="data",
                array=[r.std for r in results],
                visible=True,
            ),
            name=name,
        ),
    )


def benchmark_attention(seq_lengths: List[int], device: device | str | None) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    for seq_length in seq_lengths:
        torch.cuda.empty_cache()
        batch_size = TOTAL_TOKENS // seq_length
        if batch_size > 0:
            x = torch.randn(
                batch_size, seq_length, EMBED_DIM,
                dtype=torch.float16,
                device=device,
            )
            logging.info(f"Returned tensor shape {x.shape}")
            attn = get_multihead_dilated_attention_for_seq_length(seq_length)
            fn = partial(attention_forward, attn=attn)
            result = benchmark(fn, x)
            results.append(result)
            logging.info(f"Sequence length {seq_length}: {result}")
        else:
            logging.info(f"Batch Size 0 reached ending for loop")
            break
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    token_count = f"{ceil(TOTAL_TOKENS / 2 ** 20)}M"

    vanilla_results: List[BenchmarkResult] = []
    dilated_results: List[BenchmarkResult] = []
    multihead_dilated_results: List[BenchmarkResult] = []

    if args.benchmark_vanilla:
        logging.info(f"Benchmark vanilla attention against {token_count} tokens...")
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

    if args.benchmark_dilated:
        logging.info(f"Benchmark dilated attention against {token_count} tokens...")
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

    if args.benchmark_multihead:
        logging.info(f"Benchmark MultiHead Dilated Attention against {token_count} tokens...")
        multihead_dilated_results: List[BenchmarkResult] = benchmark_attention(DILATED_SEQ_LENGTHS, device="cuda")

    # Get current date
    current_date = datetime.date.today()

    logging.info(f"Plotting results for {token_count} tokens...")
    fig = go.Figure()
    if args.benchmark_vanilla:
        plot_results(seq_lengths=VANILLA_SEQ_LENGTHS, results=vanilla_results, name="Vanilla Attention")
    if args.benchmark_dilated:
        plot_results(seq_lengths=DILATED_SEQ_LENGTHS, results=dilated_results, name="Dilated Attention")
    if args.benchmark_multihead:
        plot_results(seq_lengths=DILATED_SEQ_LENGTHS, results=multihead_dilated_results,
                     name="MultiHead Dilated Attention")
    fig.update_layout(
        title=f"Attention Benchmark on {current_date} (Total Tokens = {token_count})",
        xaxis_title="Sequence Length",
        yaxis_title="Runtime (s)",
        xaxis_type="log",
        yaxis_type="log",
    )
    fig.write_image(os.path.join("doc", f"benchmark-{token_count}-tokens-{current_date}.png"))
