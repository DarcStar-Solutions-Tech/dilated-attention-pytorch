import logging
import os
import uuid
from enum import Enum
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
parser.add_argument('--total_tokens', type=int, default=24,
                    help='Exponent for Total tokens for benchmarking, default is 26 which is 64M(2**26) tokens')
parser.add_argument('--heads', type=int, default=4, help='Number of heads for benchmarking, default is 8')
parser.add_argument('--embed_dim', type=int, default=32, help='Embed dimension for benchmarking, default is 256')
parser.add_argument('--vanilla_seq_lengths', type=int, default=18,
                    help='End of Sequence length range for vanilla attention, default is 18 which is 2**18')
parser.add_argument('--segment_lengths', nargs='+', type=int, default=[8192, 16384, 32768, 65536],
                    help='Segment lengths for dilated attention')
parser.add_argument('--dilated_seq_lengths', type=int, default=27,
                    help='End of Sequence lengths for dilated attention, default is 27 which is 2**27 ')
parser.add_argument('--vanilla', type=bool, default=True, action=argparse.BooleanOptionalAction,
                    help='Benchmark vanilla attention')
parser.add_argument('--dilated', type=bool, default=True, action=argparse.BooleanOptionalAction,
                    help='Benchmark dilated attention')
parser.add_argument('--multihead', type=bool, default=True, action=argparse.BooleanOptionalAction,
                    help='Benchmark multihead dilated attention')
parser.add_argument('--causal', type=bool, default=False, action=argparse.BooleanOptionalAction,
                    help='Causal attention')
parser.add_argument('--permutation', type=bool, default=False, action=argparse.BooleanOptionalAction,
                    help='Benchmark permutations for heads and embed_dim')

# Parse the arguments
args = parser.parse_args()

# Generic benchmarking parameters
BATCH_SIZE: int = args.batch_size
TOTAL_TOKENS: int = 2 ** args.total_tokens  # 64M
NUM_HEADS: int = args.heads
EMBED_DIM: int = args.embed_dim
# Vanilla attention only
VANILLA_SEQ_LENGTHS: List[int] = [2 ** i for i in range(13, args.vanilla_seq_lengths)]  # 8k - 128k

# Dilated attention only
SEGMENT_LENGTHS: List[int] = args.segment_lengths  # 8k - 64k
DILATED_SEQ_LENGTHS: List[int] = [2 ** i for i in range(13, args.dilated_seq_lengths)]  # 8k - 64M

BENCHMARK_VANILLA: bool = args.vanilla
BENCHMARK_DILATED: bool = args.dilated
BENCHMARK_MULTIHEAD: bool = args.multihead
IS_CAUSAL: bool = args.causal
PERMUTATION: bool = args.permutation


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


def calculate_segments_and_dilation_rates(seq_length: int):
    segment_lengths: List[int] = []
    dilation_rates: List[int] = []
    for segment_length in SEGMENT_LENGTHS:
        # We can't use segment lengths larger than the sequence length.
        segment_length = min(segment_length, seq_length)
        exponent = segment_length // SEGMENT_LENGTHS[0] - 1
        dilation_rate = 2 ** exponent
        segment_lengths.append(segment_length)
        dilation_rates.append(dilation_rate)

    return dilation_rates, segment_lengths


class AttentionType(Enum):
    DILATED = "dilated"
    MHA = "mha"
    VANILLA = "vanilla"


def get_attention_for_seq_length(
        seq_length: int,
        device: device | str | None,
        num_heads: int = NUM_HEADS,
        embed_dim: int = EMBED_DIM,
        attention_type: AttentionType = AttentionType.DILATED,
        dtype: torch.dtype = torch.float16,
        op: xops.AttentionOp = xops.MemoryEfficientAttentionFlashAttentionOp,
) -> MultiheadDilatedAttention | DilatedAttention:
    """This is roughly how benchmarking was described in the paper, except that they
    were testing in a distributed (multi-GPU) setting.  We use a base segment
    length of 8192, and include larger segment lengths if possible.  I believe
    this is the equivalent benchmark for 1 GPU.

    Reference:
        https://arxiv.org/pdf/2307.02486.pdf, Section 3.1
    """

    dilation_rates, segment_lengths = calculate_segments_and_dilation_rates(seq_length)

    if attention_type == AttentionType.DILATED:
        return DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            op=op,
        )
    elif attention_type == AttentionType.MHA:
        return MultiheadDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            op=op,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Invalid attention type: {attention_type}")


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


def benchmark_attention(
        seq_lengths: List[int],
        device: device | str | None,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        attention_type: AttentionType = AttentionType.DILATED,
        dtype: torch.dtype = torch.float16,
        op: xops.AttentionOp = xops.MemoryEfficientAttentionFlashAttentionOp,
) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    for seq_length in seq_lengths:
        torch.cuda.empty_cache()
        batch_size = TOTAL_TOKENS // seq_length
        if batch_size > 0:
            x: torch.Tensor | None = None
            if attention_type == AttentionType.MHA:
                x = torch.randn(
                    (batch_size, seq_length, embed_dim),
                    dtype=dtype,
                    device=device,
                )
            elif (attention_type == AttentionType.DILATED) | (attention_type == AttentionType.VANILLA):
                x = torch.randn(
                    (batch_size, seq_length, num_heads, embed_dim),
                    dtype=dtype,
                    device=device,
                )

            logging.info(f"Returned tensor shape {x.shape}")

            if (attention_type == AttentionType.MHA) | (attention_type == AttentionType.DILATED):
                attn = get_attention_for_seq_length(
                    seq_length=seq_length,
                    device=device,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    attention_type=attention_type,
                    op=op,
                    dtype=dtype
                )
            else:
                attn = xops.memory_efficient_attention

            fn = partial(attention_forward, attn=attn)
            result = benchmark(fn, x)
            results.append(result)
            logging.info(f"Sequence length {seq_length}: {result}")
        else:
            logging.info(f"Batch Size 0 reached ending for loop")
            break
    return results


def bench_and_plot(
        label: str,
        token_count: str,
        seq_lengths: List[int],
        device: device | str | None,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        attention_type: AttentionType = AttentionType.VANILLA
) -> List[BenchmarkResult]:
    logging.info(f"Benchmark {label} against {token_count} tokens... with embed_dim {embed_dim} and num_heads {num_heads}")
    results: List[BenchmarkResult] = (
        benchmark_attention(
            seq_lengths=seq_lengths,
            device=device,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_type=attention_type,
        )
    )
    logging.info(f"Plotting {label} results for {token_count} tokens...")
    plot_results(seq_lengths=seq_lengths, results=results, name=f"{label}-embed_dim-{embed_dim}-heads-{num_heads}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Get current date
    current_date = datetime.date.today()

    token_count = f"{round(TOTAL_TOKENS / 2 ** 20, 2)}M"
    bench_config = {
        "batch_size": BATCH_SIZE,
        "total_tokens": TOTAL_TOKENS,
        "token_count": token_count,
        "embed_dim": EMBED_DIM,
        "heads": NUM_HEADS,
        "vanilla_seq_lengths": VANILLA_SEQ_LENGTHS,
        "segment_lengths": SEGMENT_LENGTHS,
        "dilated_seq_lengths": DILATED_SEQ_LENGTHS,
        "vanilla": BENCHMARK_VANILLA,
        "dilated": BENCHMARK_DILATED,
        "multihead": BENCHMARK_MULTIHEAD,
        "causal": IS_CAUSAL,
        "permutation": PERMUTATION,
        "date": current_date,

    }

    logging.info(f"Running benchmark with {TOTAL_TOKENS} tokens...({token_count})")
    logging.info(f"Embed Dim = {EMBED_DIM} Num Heads = {NUM_HEADS}")
    logging.info(f"Benchmarking Vanilla Attention = {BENCHMARK_VANILLA}")
    logging.info(f"Benchmarking Dilated Attention = {BENCHMARK_DILATED}")
    logging.info(f"Benchmarking MultiHead Dilated Attention = {BENCHMARK_MULTIHEAD}")
    logging.info(f"Vanilla Sequence Lengths = {VANILLA_SEQ_LENGTHS}")
    logging.info(f"Dilated Sequence Lengths = {DILATED_SEQ_LENGTHS}")
    logging.info(f"Segment Lengths = {SEGMENT_LENGTHS}")
    logging.info(f"Causal Attention = {IS_CAUSAL}")
    logging.info(f"Permutation = {PERMUTATION}")

    if not torch.cuda.is_available():  # Check if CUDA is available
        logging.info("CUDA is not available. Exiting...")
        exit()

    gpu_count = torch.cuda.device_count()  # Get total number of GPUs
    logging.info(f"Number of GPUs: {gpu_count}")

    gpu_info = {
        "gpu_count": gpu_count,
    }

    for i in range(gpu_count):
        gpu_info[f"gpu_{i}_name"] = torch.cuda.get_device_name(i)
        logging.info(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
        gpu_info[f"gpu_{i}_memory"] = round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 2)
        logging.info(f"GPU {i} memory: {round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 2)} GB")
        gpu_info[f"gpu_{i}_compute_capability"] = torch.cuda.get_device_capability(i)
        logging.info(f"GPU {i} compute capability: {torch.cuda.get_device_capability(i)}")

    bench_config.update(gpu_info)

    b_name = f"{current_date}-benchmark-{uuid.uuid4()}"
    if b_name not in os.listdir("doc"):
        os.mkdir(os.path.join("doc", b_name))

    b_dir = os.path.join("doc", b_name)

    with open(os.path.join(b_dir, f"config-{token_count}-embed_dim-{EMBED_DIM}-heads-{NUM_HEADS}.txt"), "w") as f:
        f.write(str(bench_config))

    fig = go.Figure()
    if PERMUTATION:
        # 4 is the minimum num_heads for dilated attention
        num_heads = [NUM_HEADS // 2 ** i for i in range(0, NUM_HEADS // 4)]

        # 8 * the smallest num_head is the minimum embed_dim
        embed_dims = [EMBED_DIM // 2**i for i in range(0, EMBED_DIM//(8*num_heads[-1]))]
    else:
        embed_dims = [EMBED_DIM]
        num_heads = [NUM_HEADS]

    results = []

    for embed_dim in embed_dims:  # loop over embed_dims
        for num_head in num_heads:  # loop over num_heads

            if embed_dim % num_head != 0:
                logging.info(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_head})")
                continue

            if num_head < 4:
                logging.info(f"num_heads ({num_head}) must be greater than 4")
                continue

            logging.info(f"Running benchmark with embed_dim {embed_dim} and num_heads {num_head}")

            result_set = {
                "embed_dim": embed_dim,
                "num_heads": num_head,
                "token_count": token_count,
                "vanilla": None,
                "dilated": None,
                "multihead": None,
            }

            if BENCHMARK_VANILLA:
                vanilla_results: List[BenchmarkResult] = bench_and_plot(
                    label="Vanilla Attention",
                    token_count=token_count,
                    seq_lengths=VANILLA_SEQ_LENGTHS,
                    device="cuda",
                    embed_dim=embed_dim,
                    num_heads=num_head,
                    attention_type=AttentionType.VANILLA
                )
                result_set["vanilla"] = vanilla_results

            if BENCHMARK_DILATED and embed_dim <= 128:
                dilated_results: List[BenchmarkResult] = bench_and_plot(
                    label="Dilated Attention",
                    token_count=token_count,
                    seq_lengths=DILATED_SEQ_LENGTHS,
                    device="cuda",
                    embed_dim=embed_dim,
                    num_heads=num_head,
                    attention_type=AttentionType.DILATED
                )
                result_set["dilated"] = dilated_results

            if BENCHMARK_MULTIHEAD and embed_dim//num_head <= 128:
                if embed_dim // num_head % 8 != 0:
                    logging.info(f"head_dim (embed_dim / num_heads = {embed_dim // num_head}) must be divisible by 8")

                else:
                    mha_results: List[BenchmarkResult] = bench_and_plot(
                        label="MH Dilated Attention",
                        token_count=token_count,
                        seq_lengths=DILATED_SEQ_LENGTHS,
                        device="cuda",
                        embed_dim=embed_dim,
                        num_heads=num_head,
                        attention_type=AttentionType.MHA
                    )
                    result_set["multihead"] = mha_results

            results.append(result_set)

    with open(os.path.join(b_dir, f"results-{token_count}-embed_dim-{EMBED_DIM}-heads-{NUM_HEADS}.txt"), "w") as f:
        f.write(str(results))

    fig.update_layout(
        title=f"Attention Benchmark on {current_date} <br>"
              f"(Total Tokens = {token_count}) <br>"
              f"Starting Embed Dim = {EMBED_DIM} Starting Num Heads = {NUM_HEADS}",
        title_x=0.5,
        xaxis_title="Sequence Length",
        yaxis_title="Runtime (s)",
        xaxis_type="log",
        yaxis_type="log",
    )

    fig.write_image(os.path.join(b_dir, f"tokens-{token_count}-embed_dim-{EMBED_DIM}-heads-{NUM_HEADS}.png"))
