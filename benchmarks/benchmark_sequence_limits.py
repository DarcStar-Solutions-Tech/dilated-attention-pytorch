"""
Comprehensive benchmark to find maximum sequence lengths for each implementation
"""

import gc
import time
from dataclasses import dataclass

import GPUtil
import psutil  # noqa: PLC0415
from pathlib import Path
import sys
import torch

# Import all implementations
from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
    MultiheadDilatedAttention,

# Import unified benchmark output management
sys.path.insert(0, str(Path(__file__).parent))
from core import BenchmarkOutputManager
)
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from dilated_attention_pytorch.improved_multihead_dilated_attention import (
    ImprovedMultiheadDilatedAttention,
)
from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention
from dilated_attention_pytorch.ring_multihead_dilated_attention import (
    RingMultiheadDilatedAttention,
)


@dataclass
class BenchmarkResult:
    implementation: str
    seq_len: int
    batch_size: int
    num_heads: int
    head_dim: int
    time_ms: float
    memory_gb: float
    throughput_tokens_per_sec: float
    success: bool
    error: str | None = None


class SequenceLengthBenchmark:
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.results: list[BenchmarkResult] = []

    def get_gpu_memory_info(self) -> tuple[float, float]:
        """Get current GPU memory usage in GB"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return allocated, reserved
        return 0.0, 0.0

    def get_system_memory_info(self) -> float:
        """Get system RAM usage in GB"""
        return psutil.virtual_memory().used / 1024**3

    def benchmark_implementation(
        self,
        name: str,
        module,
        seq_len: int,
        batch_size: int = 1,
        num_heads: int = 8,
        head_dim: int = 64,
        is_multihead: bool = False,
        warmup_steps: int = 3,
        benchmark_steps: int = 10,
    ) -> BenchmarkResult:
        """Benchmark a single implementation at a specific sequence length"""

        # Clear cache before starting
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        try:
            # Move module to device
            module = module.to(self.device, self.dtype)

            # Create inputs
            if is_multihead:
                embed_dim = num_heads * head_dim
                query = torch.randn(
                    batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
                )
                key = torch.randn(
                    batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
                )
                value = torch.randn(
                    batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
                )
            else:
                query = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                key = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                value = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )

            # Warmup
            for _ in range(warmup_steps):
                with torch.no_grad():
                    output = module(query, key, value)
                    if isinstance(output, tuple):
                        output = output[0]

            # Get initial memory
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_mem, _ = self.get_gpu_memory_info()

            # Benchmark
            times = []
            for _ in range(benchmark_steps):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                start_time = time.perf_counter()

                with torch.no_grad():
                    output = module(query, key, value)
                    if isinstance(output, tuple):
                        output = output[0]

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                times.append(end_time - start_time)

            # Get peak memory
            if self.device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                memory_used = peak_mem - start_mem
            else:
                memory_used = 0.0

            # Calculate metrics
            avg_time = sum(times) / len(times)
            time_ms = avg_time * 1000
            tokens_processed = batch_size * seq_len
            throughput = tokens_processed / avg_time

            # Cleanup
            del query, key, value, output
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            return BenchmarkResult(
                implementation=name,
                seq_len=seq_len,
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                time_ms=time_ms,
                memory_gb=memory_used,
                throughput_tokens_per_sec=throughput,
                success=True,
            )

        except Exception as e:
            # Cleanup on error
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            return BenchmarkResult(
                implementation=name,
                seq_len=seq_len,
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                time_ms=0,
                memory_gb=0,
                throughput_tokens_per_sec=0,
                success=False,
                error=str(e),
            )

    def find_max_sequence_length(
        self,
        name: str,
        module_factory,
        seq_lengths: list[int],
        batch_size: int = 1,
        num_heads: int = 8,
        head_dim: int = 64,
        is_multihead: bool = False,
    ) -> int:
        """Find maximum working sequence length for an implementation"""

        max_working_len = 0

        for seq_len in seq_lengths:
            print(f"  Testing {name} at seq_len={seq_len}...", end="", flush=True)

            # Create fresh module for each test
            try:
                module = module_factory(seq_len)
            except Exception as e:
                print(f" ✗ Module creation failed: {e}")
                break

            result = self.benchmark_implementation(
                name, module, seq_len, batch_size, num_heads, head_dim, is_multihead
            )

            self.results.append(result)

            if result.success:
                max_working_len = seq_len
                print(
                    f" ✓ {result.time_ms:.1f}ms, {result.memory_gb:.2f}GB, {result.throughput_tokens_per_sec:.0f} tok/s"
                )
            else:
                print(f" ✗ {result.error}")
                break

            # Cleanup
            del module
            gc.collect()

        return max_working_len

    def run_comprehensive_benchmark(self):
        """Run benchmarks on all implementations"""

        # Test sequence lengths - exponentially increasing
        seq_lengths = [
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            524288,
            1048576,
        ]

        # Common parameters
        batch_size = 1
        num_heads = 8
        head_dim = 64
        embed_dim = num_heads * head_dim
        dropout = 0.0

        print("=" * 80)
        print("SEQUENCE LENGTH LIMIT BENCHMARK")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")
        print(f"Batch size: {batch_size}")
        print(f"Num heads: {num_heads}")
        print(f"Head dim: {head_dim}")

        if self.device.type == "cuda":
            gpu = GPUtil.getGPUs()[0]
            print(f"GPU: {gpu.name}")
            print(f"GPU Memory: {gpu.memoryTotal}MB")

        print("\nTesting sequence lengths:", seq_lengths)
        print("=" * 80)

        # Define implementations with factories
        implementations = [
            # Core implementations
            (
                "DilatedAttention",
                lambda seq_len: DilatedAttention(
                    segment_lengths=[
                        min(1024, seq_len // 4),
                        min(2048, seq_len // 2),
                        min(4096, seq_len),
                    ],
                    dilation_rates=[1, 2, 4],
                    attention_dropout=dropout,
                ),
                False,
            ),
            (
                "ImprovedDilatedAttention",
                lambda seq_len: ImprovedDilatedAttention(
                    segment_lengths=[
                        min(1024, seq_len // 4),
                        min(2048, seq_len // 2),
                        min(4096, seq_len),
                    ],
                    dilation_rates=[1, 2, 4],
                    dropout=dropout,
                ),
                False,
            ),
            (
                "RingDilatedAttention",
                lambda seq_len: RingDilatedAttention(
                    segment_lengths=[
                        min(1024, seq_len // 4),
                        min(2048, seq_len // 2),
                        min(4096, seq_len),
                    ],
                    dilation_rates=[1, 2, 4],
                    dropout=dropout,
                    ring_size=1,
                ),
                False,
            ),
            (
                "BlockSparseRing_10%",
                lambda seq_len: BlockSparseRingDilatedAttention(
                    segment_lengths=[
                        min(1024, seq_len // 4),
                        min(2048, seq_len // 2),
                        min(4096, seq_len),
                    ],
                    dilation_rates=[1, 2, 4],
                    sparse_config=SparsePatternConfig(
                        pattern_type="dilated_sparse",
                        sparsity_ratio=0.1,
                        block_size=min(128, seq_len // 16),
                    ),
                    dropout=dropout,
                    ring_size=1,
                ),
                False,
            ),
            (
                "BlockSparseRing_25%",
                lambda seq_len: BlockSparseRingDilatedAttention(
                    segment_lengths=[
                        min(1024, seq_len // 4),
                        min(2048, seq_len // 2),
                        min(4096, seq_len),
                    ],
                    dilation_rates=[1, 2, 4],
                    sparse_config=SparsePatternConfig(
                        pattern_type="dilated_sparse",
                        sparsity_ratio=0.25,
                        block_size=min(128, seq_len // 16),
                    ),
                    dropout=dropout,
                    ring_size=1,
                ),
                False,
            ),
            # Multihead implementations
            (
                "MultiheadDilatedAttention",
                lambda seq_len: MultiheadDilatedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=[
                        min(1024, seq_len // 4),
                        min(2048, seq_len // 2),
                        min(4096, seq_len),
                    ],
                    dilation_rates=[1, 2, 4],
                    dropout=dropout,
                ),
                True,
            ),
            (
                "ImprovedMultiheadDilated",
                lambda seq_len: ImprovedMultiheadDilatedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=[
                        min(1024, seq_len // 4),
                        min(2048, seq_len // 2),
                        min(4096, seq_len),
                    ],
                    dilation_rates=[1, 2, 4],
                    dropout=dropout,
                ),
                True,
            ),
            (
                "RingMultiheadDilated",
                lambda seq_len: RingMultiheadDilatedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=[
                        min(1024, seq_len // 4),
                        min(2048, seq_len // 2),
                        min(4096, seq_len),
                    ],
                    dilation_rates=[1, 2, 4],
                    dropout=dropout,
                    ring_size=1,
                ),
                True,
            ),
        ]

        # Run benchmarks
        max_lengths = {}

        for name, factory, is_multihead in implementations:
            print(f"\n{name}:")
            max_len = self.find_max_sequence_length(
                name,
                factory,
                seq_lengths,
                batch_size,
                num_heads,
                head_dim,
                is_multihead,
            )
            max_lengths[name] = max_len

            # Give GPU a break between implementations
            time.sleep(1)

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY: Maximum Sequence Lengths")
        print("=" * 80)

        for name, max_len in sorted(
            max_lengths.items(), key=lambda x: x[1], reverse=True
        ):
            if max_len > 0:
                print(f"{name:30} {max_len:>10,} tokens")
            else:
                print(f"{name:30} {'Failed':>10}")

        # Print use case recommendations
        self.print_use_case_recommendations(max_lengths)

    def print_use_case_recommendations(self, max_lengths: dict[str, int]):
        """Print recommendations for different use cases"""

        print("\n" + "=" * 80)
        print("USE CASE RECOMMENDATIONS")
        print("=" * 80)

        # Group by characteristics
        short_seq = [(k, v) for k, v in max_lengths.items() if 0 < v <= 8192]
        medium_seq = [(k, v) for k, v in max_lengths.items() if 8192 < v <= 65536]
        long_seq = [(k, v) for k, v in max_lengths.items() if v > 65536]

        print("\n1. SHORT SEQUENCES (≤8K tokens) - Real-time applications:")
        print("   Best for: Chatbots, code completion, real-time translation")
        for name, max_len in sorted(short_seq, key=lambda x: x[1], reverse=True):
            # Find performance at 4K
            perf = next(
                (
                    r
                    for r in self.results
                    if r.implementation == name and r.seq_len == 4096 and r.success
                ),
                None,
            )
            if perf:
                print(
                    f"   - {name}: {perf.time_ms:.1f}ms @ 4K tokens ({perf.throughput_tokens_per_sec:.0f} tok/s)"
                )

        print("\n2. MEDIUM SEQUENCES (8K-64K tokens) - Document processing:")
        print("   Best for: Document summarization, long-form generation, RAG")
        for name, max_len in sorted(medium_seq, key=lambda x: x[1], reverse=True):
            # Find performance at 32K
            perf = next(
                (
                    r
                    for r in self.results
                    if r.implementation == name and r.seq_len == 32768 and r.success
                ),
                None,
            )
            if perf:
                print(
                    f"   - {name}: {perf.time_ms:.1f}ms @ 32K tokens ({perf.throughput_tokens_per_sec:.0f} tok/s)"
                )

        print("\n3. LONG SEQUENCES (>64K tokens) - Large document processing:")
        print("   Best for: Book analysis, codebase understanding, research papers")
        for name, max_len in sorted(long_seq, key=lambda x: x[1], reverse=True):
            print(f"   - {name}: up to {max_len:,} tokens")
            # Find largest successful benchmark
            largest = max(
                (r for r in self.results if r.implementation == name and r.success),
                key=lambda x: x.seq_len,
                default=None,
            )
            if largest:
                print(
                    f"     Performance: {largest.time_ms:.1f}ms @ {largest.seq_len:,} tokens ({largest.throughput_tokens_per_sec:.0f} tok/s)"
                )

        print("\n4. MEMORY EFFICIENCY RANKING (at 32K tokens):")
        mem_results = [
            (r.implementation, r.memory_gb)
            for r in self.results
            if r.seq_len == 32768 and r.success
        ]
        for name, mem in sorted(mem_results, key=lambda x: x[1]):
            print(f"   - {name}: {mem:.2f}GB")

        print("\n5. SPEED RANKING (at 8K tokens):")
        speed_results = [
            (r.implementation, r.throughput_tokens_per_sec)
            for r in self.results
            if r.seq_len == 8192 and r.success
        ]
        for name, throughput in sorted(speed_results, key=lambda x: x[1], reverse=True):
            print(f"   - {name}: {throughput:.0f} tokens/sec")


if __name__ == "__main__":
    benchmark = SequenceLengthBenchmark()
    benchmark.run_comprehensive_benchmark()

    # Use unified benchmark output management
    output_manager = BenchmarkOutputManager(
        benchmark_type="sequence-limits",
        parameters={}
    )
    
    # Add results
    output_manager.add_result("results", results)
    
    # Save results
    output_paths = output_manager.save_results()
    print(f"\nResults saved to:")
    for path_type, path in output_paths.items():
        print(f"  {path_type}: {path}")
