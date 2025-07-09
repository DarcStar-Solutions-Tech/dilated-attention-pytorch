"""Comprehensive verification script for all implementations."""

import sys
from typing import Dict, Any

import torch
import torch.distributed as dist

from core.utils import (
    cleanup_memory,
    get_rank,
    get_world_size,
)


class ImplementationVerifier:
    """Verify all dilated attention implementations."""

    def __init__(self):
        """Initialize verifier."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.results = []

    def verify_imports(self) -> Dict[str, bool]:
        """Verify all implementations can be imported."""
        implementations = {
            "DilatedAttention": "dilated_attention_pytorch.DilatedAttention",
            "MultiheadDilatedAttention": "dilated_attention_pytorch.MultiheadDilatedAttention",
            "ImprovedDilatedAttention": "dilated_attention_pytorch.ImprovedDilatedAttention",
            "ImprovedMultiheadDilatedAttention": "dilated_attention_pytorch.ImprovedMultiheadDilatedAttention",
            "RingDilatedAttentionProduction": "dilated_attention_pytorch.RingDilatedAttentionProduction",
            "BlockSparseRingDilatedAttention": "dilated_attention_pytorch.BlockSparseRingDilatedAttention",
        }

        results = {}
        for name, module_path in implementations.items():
            try:
                module_name, class_name = module_path.rsplit(".", 1)
                module = __import__(module_name, fromlist=[class_name])
                _ = getattr(module, class_name)
                results[name] = True
                print(f"✓ {name} imported successfully")
            except Exception as e:
                results[name] = False
                print(f"✗ {name} import failed: {e}")

        return results

    def verify_device_support(self) -> Dict[str, Any]:
        """Verify device support."""
        results = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
            "current_device": str(self.device),
        }

        if torch.cuda.is_available():
            results["cuda_version"] = torch.version.cuda
            results["cudnn_version"] = torch.backends.cudnn.version()
            results["device_name"] = torch.cuda.get_device_name(0)
            results["device_capability"] = torch.cuda.get_device_capability(0)

        print("\nDevice Support:")
        for key, value in results.items():
            print(f"  {key}: {value}")

        return results

    def verify_distributed_setup(self) -> Dict[str, Any]:
        """Verify distributed setup."""
        results = {
            "distributed_available": dist.is_available(),
            "nccl_available": dist.is_nccl_available()
            if dist.is_available()
            else False,
            "gloo_available": dist.is_gloo_available()
            if dist.is_available()
            else False,
            "initialized": dist.is_initialized() if dist.is_available() else False,
        }

        if dist.is_initialized():
            results["rank"] = get_rank()
            results["world_size"] = get_world_size()
            results["backend"] = dist.get_backend()

        print("\nDistributed Setup:")
        for key, value in results.items():
            print(f"  {key}: {value}")

        return results

    def verify_dtype_support(self) -> Dict[str, bool]:
        """Verify data type support."""
        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        results = {}
        print("\nData Type Support:")

        for name, dtype in dtypes.items():
            try:
                # Test tensor creation
                x = torch.randn(2, 2, dtype=dtype, device=self.device)
                # Test basic operation
                _ = x @ x.T
                results[name] = True
                print(f"  ✓ {name} supported")
            except Exception as e:
                results[name] = False
                print(f"  ✗ {name} not supported: {e}")

        return results

    def verify_memory(self) -> Dict[str, float]:
        """Verify memory availability."""
        results = {}

        if torch.cuda.is_available():
            cleanup_memory()

            # Get memory info for each device
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_memory / 1024**2
                allocated_mb = torch.cuda.memory_allocated(i) / 1024**2
                reserved_mb = torch.cuda.memory_reserved(i) / 1024**2
                free_mb = total_mb - allocated_mb

                results[f"gpu_{i}_total_mb"] = total_mb
                results[f"gpu_{i}_allocated_mb"] = allocated_mb
                results[f"gpu_{i}_reserved_mb"] = reserved_mb
                results[f"gpu_{i}_free_mb"] = free_mb

                print(f"\nGPU {i} Memory:")
                print(f"  Total: {total_mb:,.0f} MB")
                print(f"  Allocated: {allocated_mb:,.0f} MB")
                print(f"  Reserved: {reserved_mb:,.0f} MB")
                print(f"  Free: {free_mb:,.0f} MB")
        else:
            print("\nMemory: No GPU available")

        return results

    def verify_basic_operations(self) -> Dict[str, bool]:
        """Verify basic operations work correctly."""
        results = {}

        print("\nBasic Operations:")

        # Test configurations
        test_configs = [
            {"seq_len": 1024, "batch_size": 1, "num_heads": 8, "head_dim": 64},
            {"seq_len": 2048, "batch_size": 2, "num_heads": 16, "head_dim": 64},
        ]

        from dilated_attention_pytorch import ImprovedDilatedAttention

        for i, config in enumerate(test_configs):
            try:
                model = ImprovedDilatedAttention(
                    segment_lengths=[512, 1024],
                    dilation_rates=[1, 2],
                    dropout=0.0,
                ).to(self.device)

                # Generate test data
                shape = (
                    config["batch_size"],
                    config["seq_len"],
                    config["num_heads"],
                    config["head_dim"],
                )
                q = torch.randn(*shape, device=self.device, dtype=self.dtype)
                k = torch.randn(*shape, device=self.device, dtype=self.dtype)
                v = torch.randn(*shape, device=self.device, dtype=self.dtype)

                # Forward pass
                output = model(q, k, v)

                # Verify output shape
                assert output.shape == q.shape, (
                    f"Output shape mismatch: {output.shape} vs {q.shape}"
                )

                results[f"config_{i}"] = True
                print(
                    f"  ✓ Config {i}: seq_len={config['seq_len']}, batch={config['batch_size']}"
                )

            except Exception as e:
                results[f"config_{i}"] = False
                print(f"  ✗ Config {i} failed: {e}")

        return results

    def run_all_verifications(self) -> Dict[str, Any]:
        """Run all verification tests."""
        print("=" * 80)
        print("DILATED ATTENTION VERIFICATION")
        print("=" * 80)

        all_results = {}

        # Import verification
        all_results["imports"] = self.verify_imports()

        # Device support
        all_results["device_support"] = self.verify_device_support()

        # Distributed setup
        all_results["distributed_setup"] = self.verify_distributed_setup()

        # Data type support
        all_results["dtype_support"] = self.verify_dtype_support()

        # Memory verification
        all_results["memory"] = self.verify_memory()

        # Basic operations
        all_results["basic_operations"] = self.verify_basic_operations()

        # Summary
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)

        total_tests = 0
        passed_tests = 0

        for category, results in all_results.items():
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, bool):
                        total_tests += 1
                        if value:
                            passed_tests += 1

        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests / total_tests * 100:.1f}%")

        return all_results


def main():
    """Main entry point."""
    verifier = ImplementationVerifier()
    results = verifier.run_all_verifications()

    # Exit with error code if any tests failed
    for category, category_results in results.items():
        if isinstance(category_results, dict):
            for key, value in category_results.items():
                if isinstance(value, bool) and not value:
                    sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
