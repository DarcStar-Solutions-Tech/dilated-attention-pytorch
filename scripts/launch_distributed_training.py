#!/usr/bin/env python3
"""
Launcher script for distributed dilated attention training.

This script provides an easy way to launch distributed training with optimal
configurations for different hardware setups and model sizes.

Usage:
    python scripts/launch_distributed_training.py --model_size medium --num_gpus 8
    python scripts/launch_distributed_training.py --model_size large --num_nodes 2 --num_gpus_per_node 8
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def create_deepspeed_config(
    output_path: str,
    model_size: str,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    zero_stage: int,
    cpu_offload: bool,
    use_fp16: bool,
):
    """Create optimized DeepSpeed configuration."""

    # Base configuration
    config = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": 1000,
                "total_num_steps": 100000,
            },
        },
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }

    # Add offloading for large models or memory constraints
    if cpu_offload or model_size == "large":
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

        if zero_stage >= 3:
            config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }

    # Mixed precision configuration
    if use_fp16:
        config["fp16"] = {
            "enabled": True,
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
    else:
        config["bf16"] = {"enabled": True}

    # Activation checkpointing for large models
    if model_size in ["large", "xl"]:
        config["activation_checkpointing"] = {
            "partition_activations": False,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "synchronize_checkpoint_boundary": False,
        }

    # Save configuration
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    return output_path


def get_model_config(model_size: str, max_seq_len: int):
    """Get model configuration for different sizes."""

    configs = {
        "tiny": {
            "embed_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "segment_lengths": [1024, 2048],
            "dilation_rates": [1, 2],
            "vocab_size": 32000,
        },
        "small": {
            "embed_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "segment_lengths": [2048, 4096, 8192],
            "dilation_rates": [1, 2, 4],
            "vocab_size": 50000,
        },
        "medium": {
            "embed_dim": 1024,
            "num_heads": 16,
            "num_layers": 24,
            "segment_lengths": [2048, 4096, 8192, 16384],
            "dilation_rates": [1, 2, 4, 8],
            "vocab_size": 50000,
        },
        "large": {
            "embed_dim": 2048,
            "num_heads": 32,
            "num_layers": 24,
            "segment_lengths": [2048, 4096, 8192, 16384, 32768],
            "dilation_rates": [1, 2, 4, 8, 16],
            "vocab_size": 50000,
        },
        "xl": {
            "embed_dim": 2560,
            "num_heads": 32,
            "num_layers": 32,
            "segment_lengths": [2048, 4096, 8192, 16384, 32768],
            "dilation_rates": [1, 2, 4, 8, 16],
            "vocab_size": 50000,
        },
    }

    config = configs.get(model_size, configs["medium"])
    config["max_seq_len"] = max_seq_len

    return config


def get_training_config(model_size: str, num_gpus: int, hardware_type: str = "a100"):
    """Get optimal training configuration for model size and hardware."""

    # Base configurations optimized for different model sizes
    base_configs = {
        "tiny": {
            "micro_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-4,
            "zero_stage": 1,
            "cpu_offload": False,
            "use_fp16": True,
        },
        "small": {
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": 3e-4,
            "zero_stage": 2,
            "cpu_offload": False,
            "use_fp16": True,
        },
        "medium": {
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "zero_stage": 2,
            "cpu_offload": num_gpus < 8,  # CPU offload for smaller setups
            "use_fp16": True,
        },
        "large": {
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 6e-5,
            "zero_stage": 3,
            "cpu_offload": True,
            "use_fp16": hardware_type != "h100",  # Use BF16 on H100
        },
        "xl": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 3e-5,
            "zero_stage": 3,
            "cpu_offload": True,
            "use_fp16": hardware_type != "h100",
        },
    }

    config = base_configs.get(model_size, base_configs["medium"])

    # Calculate total batch size
    config["train_batch_size"] = (
        config["micro_batch_size"] * config["gradient_accumulation_steps"] * num_gpus
    )

    # Adjust for H100 optimizations
    if hardware_type == "h100":
        config["use_fp16"] = False  # Use BF16 instead
        config["micro_batch_size"] *= 2  # H100 has more memory

    return config


def create_hostfile(nodes: list, gpus_per_node: int, output_path: str):
    """Create hostfile for multi-node training."""

    with open(output_path, "w") as f:
        for node in nodes:
            f.write(f"{node} slots={gpus_per_node}\n")

    return output_path


def build_command(
    script_path: str,
    model_config: dict,
    training_config: dict,
    num_nodes: int,
    num_gpus_per_node: int,
    master_addr: str,
    master_port: int,
    deepspeed_config: str,
    output_dir: str,
    launcher: str = "torchrun",
    hostfile: str = None,
    **kwargs,
):
    """Build the command to launch distributed training."""

    total_gpus = num_nodes * num_gpus_per_node

    # Base script arguments
    script_args = [
        "--model_size",
        kwargs.get("model_size", "medium"),
        "--vocab_size",
        str(model_config["vocab_size"]),
        "--max_seq_len",
        str(model_config["max_seq_len"]),
        "--num_epochs",
        str(kwargs.get("num_epochs", 10)),
        "--batch_size",
        str(training_config["micro_batch_size"]),
        "--learning_rate",
        str(training_config["learning_rate"]),
        "--gradient_accumulation_steps",
        str(training_config["gradient_accumulation_steps"]),
        "--checkpoint_dir",
        output_dir,
        "--use_deepspeed",
        "--deepspeed_config",
        deepspeed_config,
        "--zero_stage",
        str(training_config["zero_stage"]),
    ]

    if training_config["cpu_offload"]:
        script_args.append("--cpu_offload")

    if kwargs.get("use_wandb", False):
        script_args.append("--use_wandb")

    # Build launcher command
    if launcher == "torchrun":
        if num_nodes == 1:
            # Single node
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={num_gpus_per_node}",
                script_path,
            ] + script_args
        else:
            # Multi-node
            cmd = [
                "torchrun",
                f"--nproc_per_node={num_gpus_per_node}",
                f"--nnodes={num_nodes}",
                "--node_rank=0",  # This should be set per node
                f"--master_addr={master_addr}",
                f"--master_port={master_port}",
                script_path,
            ] + script_args

    elif launcher == "deepspeed":
        cmd = ["deepspeed"]

        if hostfile:
            cmd.extend(["--hostfile", hostfile])
        else:
            cmd.extend([f"--num_gpus={num_gpus_per_node}"])

        if num_nodes > 1:
            cmd.extend([f"--num_nodes={num_nodes}"])

        cmd.extend([script_path] + script_args)

    else:
        raise ValueError(f"Unknown launcher: {launcher}")

    return cmd


def setup_environment():
    """Setup environment variables for optimal performance."""

    env_vars = {
        # NCCL optimizations
        "NCCL_TREE_THRESHOLD": "0",
        "NCCL_IB_DISABLE": "0",
        "NCCL_DEBUG": "WARN",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        # PyTorch optimizations
        "TORCH_DIST_DEFAULT_TIMEOUT": "1800",  # 30 minutes
        "OMP_NUM_THREADS": "1",
        # CUDA optimizations
        "CUDA_LAUNCH_BLOCKING": "0",
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    print("Environment variables set for optimal performance")


def main():
    parser = argparse.ArgumentParser(
        description="Launch distributed dilated attention training"
    )

    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["tiny", "small", "medium", "large", "xl"],
        help="Model size preset",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=16384, help="Maximum sequence length"
    )

    # Hardware configuration
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument(
        "--num_gpus",
        "--num_gpus_per_node",
        type=int,
        default=8,
        dest="num_gpus_per_node",
        help="Number of GPUs per node",
    )
    parser.add_argument(
        "--hardware_type",
        type=str,
        default="a100",
        choices=["v100", "a100", "h100"],
        help="Hardware type for optimizations",
    )

    # Training configuration
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )

    # Distributed configuration
    parser.add_argument(
        "--launcher",
        type=str,
        default="torchrun",
        choices=["torchrun", "deepspeed"],
        help="Launcher to use",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Master node address for multi-node training",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for multi-node training",
    )
    parser.add_argument(
        "--hostfile", type=str, default=None, help="Hostfile for multi-node training"
    )

    # Advanced options
    parser.add_argument(
        "--dry_run", action="store_true", help="Print command without executing"
    )
    parser.add_argument(
        "--custom_script", type=str, default=None, help="Path to custom training script"
    )

    args = parser.parse_args()

    # Setup environment
    setup_environment()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get configurations
    model_config = get_model_config(args.model_size, args.max_seq_len)
    training_config = get_training_config(
        args.model_size, args.num_nodes * args.num_gpus_per_node, args.hardware_type
    )

    print("Model Configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")

    print("\nTraining Configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")

    # Create DeepSpeed config
    deepspeed_config_path = os.path.join(args.output_dir, "ds_config.json")
    create_deepspeed_config(
        output_path=deepspeed_config_path,
        model_size=args.model_size,
        train_batch_size=training_config["train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        zero_stage=training_config["zero_stage"],
        cpu_offload=training_config["cpu_offload"],
        use_fp16=training_config["use_fp16"],
    )

    print(f"\nDeepSpeed config created: {deepspeed_config_path}")

    # Create hostfile if needed
    hostfile_path = None
    if args.hostfile:
        hostfile_path = args.hostfile
    elif args.num_nodes > 1:
        # Create default hostfile (user should modify)
        hostfile_path = os.path.join(args.output_dir, "hostfile")
        with open(hostfile_path, "w") as f:
            f.write("# Edit this hostfile with your actual node addresses\n")
            for i in range(args.num_nodes):
                f.write(f"node{i} slots={args.num_gpus_per_node}\n")
        print(f"Default hostfile created: {hostfile_path}")
        print("Please edit with actual node addresses before running!")

    # Determine script path
    if args.custom_script:
        script_path = args.custom_script
    else:
        # Use the distributed training example
        script_dir = Path(__file__).parent.parent
        script_path = script_dir / "examples" / "distributed_training_example.py"

        if not script_path.exists():
            print(f"Error: Training script not found at {script_path}")
            print("Please specify --custom_script or ensure the example script exists")
            sys.exit(1)

    # Build command
    cmd = build_command(
        script_path=str(script_path),
        model_config=model_config,
        training_config=training_config,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node,
        master_addr=args.master_addr,
        master_port=args.master_port,
        deepspeed_config=deepspeed_config_path,
        output_dir=args.output_dir,
        launcher=args.launcher,
        hostfile=hostfile_path,
        model_size=args.model_size,
        num_epochs=args.num_epochs,
        use_wandb=args.use_wandb,
    )

    print("\nLaunch Command:")
    print(" ".join(cmd))

    if args.dry_run:
        print("\nDry run - command not executed")
        return

    print("\nStarting distributed training...")
    print(f"Model: {args.model_size}")
    print(f"Nodes: {args.num_nodes}")
    print(f"GPUs per node: {args.num_gpus_per_node}")
    print(f"Total GPUs: {args.num_nodes * args.num_gpus_per_node}")
    print(f"Output directory: {args.output_dir}")

    # Execute command
    try:
        result = subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
