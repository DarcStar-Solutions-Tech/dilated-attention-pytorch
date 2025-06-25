#!/usr/bin/env python3
"""
Comprehensive example for distributed training with advanced dilated attention.

This script demonstrates how to use the advanced distributed dilated attention
implementation with DeepSpeed, model parallelism, and sequence parallelism.

Usage:
    # Single node, 8 GPUs
    torchrun --standalone --nproc_per_node=8 distributed_training_example.py

    # Multi-node setup (2 nodes, 8 GPUs each)
    # Node 0:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
             --master_addr="192.168.1.1" --master_port=29500 \
             distributed_training_example.py

    # Node 1:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
             --master_addr="192.168.1.1" --master_port=29500 \
             distributed_training_example.py

    # With DeepSpeed launcher
    deepspeed --num_gpus=8 distributed_training_example.py \
              --deepspeed_config=ds_config.json
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dilated_attention_pytorch.improved_distributed_dilated_attention import (
    DistributedImprovedMultiheadDilatedAttention,
    DeepSpeedDilatedAttentionEngine,
    create_distributed_model,
    get_recommended_config
)

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class DistributedTrainer:
    """
    Distributed trainer for dilated attention models with advanced optimizations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        use_deepspeed: bool = False,
        use_wandb: bool = False,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config or {}
        self.use_deepspeed = use_deepspeed and HAS_DEEPSPEED
        self.use_wandb = use_wandb and HAS_WANDB
        self.checkpoint_dir = checkpoint_dir
        
        # Distributed setup
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.is_main_process = self.rank == 0
        
        # Setup device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cpu')
        
        # Mixed precision setup
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp and not self.use_deepspeed else None
        
        # Initialize DeepSpeed if enabled
        if self.use_deepspeed:
            self._setup_deepspeed()
        else:
            self.model = self.model.to(self.device)
            if dist.is_initialized():
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.local_rank]
                )
        
        # Setup logging
        if self.use_wandb and self.is_main_process:
            wandb.init(
                project=self.config.get('project_name', 'dilated-attention'),
                name=self.config.get('run_name', 'distributed-training'),
                config=self.config
            )
        
        # Create checkpoint directory
        if self.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _setup_deepspeed(self):
        """Setup DeepSpeed optimization."""
        if not self.use_deepspeed:
            return
        
        # Create DeepSpeed config if not provided
        ds_config_path = self.config.get('deepspeed_config', None)
        if ds_config_path is None:
            ds_config_path = os.path.join(self.checkpoint_dir, 'ds_config.json')
            DeepSpeedDilatedAttentionEngine.create_config_file(
                output_path=ds_config_path,
                train_batch_size=self.config.get('train_batch_size', 16),
                gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
                learning_rate=self.config.get('learning_rate', 1e-4),
                zero_stage=self.config.get('zero_stage', 2),
                cpu_offload=self.config.get('cpu_offload', False),
                use_fp16=self.config.get('use_fp16', True)
            )
        
        # Initialize DeepSpeed
        self.model, self.optimizer, _, self.lr_scheduler = (
            DeepSpeedDilatedAttentionEngine.initialize_deepspeed(
                model=self.model,
                config_path=ds_config_path
            )
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = len(self.train_dataloader)
        
        # Set epoch for distributed sampler
        if hasattr(self.train_dataloader.sampler, 'set_epoch'):
            self.train_dataloader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(self.train_dataloader):
            start_time = time.time()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)
            
            # Forward pass
            if self.use_deepspeed:
                # DeepSpeed handles mixed precision automatically
                logits = self.model(input_ids, is_causal=True)
                
                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                
                # Backward pass with DeepSpeed
                self.model.backward(loss)
                self.model.step()
                
            else:
                # Standard training loop
                if self.use_amp and self.scaler:
                    with autocast():
                        logits = self.model(input_ids, is_causal=True)
                        
                        # Calculate loss
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100
                        )
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision
                    logits = self.model(input_ids, is_causal=True)
                    
                    # Calculate loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Learning rate scheduling
                if self.lr_scheduler:
                    self.lr_scheduler.step()
            
            # Statistics
            batch_loss = loss.item()
            batch_tokens = input_ids.numel()
            total_loss += batch_loss
            total_tokens += batch_tokens
            
            # Logging
            if step % self.config.get('log_interval', 100) == 0:
                elapsed_time = time.time() - start_time
                tokens_per_sec = batch_tokens / elapsed_time
                
                if self.is_main_process:
                    print(f"Epoch {epoch}, Step {step}/{num_batches}, "
                          f"Loss: {batch_loss:.4f}, "
                          f"Tokens/sec: {tokens_per_sec:.0f}")
                    
                    if self.use_wandb:
                        wandb.log({
                            'train/loss': batch_loss,
                            'train/tokens_per_sec': tokens_per_sec,
                            'train/learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0,
                            'epoch': epoch,
                            'step': step
                        })
            
            # Memory monitoring
            if step % self.config.get('memory_log_interval', 500) == 0 and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                
                if self.is_main_process:
                    print(f"GPU Memory - Allocated: {memory_allocated:.1f}GB, "
                          f"Reserved: {memory_reserved:.1f}GB")
        
        avg_loss = total_loss / num_batches
        tokens_per_sec = total_tokens / (time.time() - start_time) if num_batches > 0 else 0
        
        return {
            'loss': avg_loss,
            'tokens_per_sec': tokens_per_sec
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        if not self.val_dataloader:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch.get('labels', input_ids).to(self.device)
                
                if self.use_amp:
                    with autocast():
                        logits = self.model(input_ids, is_causal=True)
                        
                        # Calculate loss
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100
                        )
                else:
                    logits = self.model(input_ids, is_causal=True)
                    
                    # Calculate loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        if self.is_main_process:
            print(f"Validation Loss: {avg_loss:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'val/loss': avg_loss,
                    'epoch': epoch
                })
        
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        if not self.is_main_process:
            return
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        
        if self.use_deepspeed:
            # DeepSpeed handles checkpointing
            self.model.save_checkpoint(self.checkpoint_dir, tag=f'epoch_{epoch}')
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'metrics': metrics,
                'config': self.config
            }
            
            torch.save(checkpoint, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            if self.is_main_process:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate(epoch)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Save checkpoint
            if self.config.get('save_checkpoints', True):
                if epoch % self.config.get('checkpoint_interval', 5) == 0:
                    self.save_checkpoint(epoch, metrics)
            
            # Save best model
            current_val_loss = val_metrics.get('val_loss', float('inf'))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                if self.is_main_process:
                    best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                    if self.use_deepspeed:
                        self.model.save_checkpoint(self.checkpoint_dir, tag='best')
                    else:
                        torch.save(self.model.state_dict(), best_checkpoint_path)
                    print(f"New best model saved with val_loss: {best_val_loss:.4f}")


def create_dummy_dataset(vocab_size: int, seq_len: int, num_samples: int = 1000):
    """Create a dummy dataset for testing."""
    class DummyDataset:
        def __init__(self):
            self.data = []
            for _ in range(num_samples):
                input_ids = torch.randint(0, vocab_size, (seq_len,))
                self.data.append({'input_ids': input_ids})
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return DummyDataset()


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size
        )
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Distributed Dilated Attention Training')
    parser.add_argument('--model_size', type=str, default='medium', 
                       choices=['small', 'medium', 'large'], help='Model size')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--max_seq_len', type=int, default=16384, help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                       help='Gradient accumulation steps')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                       help='Checkpoint directory')
    parser.add_argument('--use_deepspeed', action='store_true', help='Use DeepSpeed optimization')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--deepspeed_config', type=str, default=None, 
                       help='DeepSpeed configuration file')
    parser.add_argument('--cpu_offload', action='store_true', help='Offload to CPU memory')
    parser.add_argument('--zero_stage', type=int, default=2, choices=[1, 2, 3], 
                       help='DeepSpeed ZeRO stage')
    
    args = parser.parse_args()
    
    # Setup distributed training
    is_distributed = setup_distributed()
    
    # Get model configuration
    world_size = dist.get_world_size() if is_distributed else 1
    config = get_recommended_config(args.model_size, world_size)
    
    # Override with command line arguments
    config.update({
        'vocab_size': args.vocab_size,
        'max_seq_len': args.max_seq_len,
        'train_batch_size': args.batch_size * world_size * args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'use_deepspeed': args.use_deepspeed and HAS_DEEPSPEED,
        'deepspeed_config': args.deepspeed_config,
        'cpu_offload': args.cpu_offload,
        'zero_stage': args.zero_stage,
        'use_amp': True,
        'use_fp16': True,
        'project_name': 'dilated-attention-distributed',
        'run_name': f'{args.model_size}-{world_size}gpu'
    })
    
    rank = dist.get_rank() if is_distributed else 0
    if rank == 0:
        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Create model
    model = create_distributed_model(**config)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel created with {total_params:,} parameters")
    
    # Create datasets
    train_dataset = create_dummy_dataset(
        vocab_size=args.vocab_size, 
        seq_len=args.max_seq_len,
        num_samples=10000
    )
    val_dataset = create_dummy_dataset(
        vocab_size=args.vocab_size,
        seq_len=args.max_seq_len, 
        num_samples=1000
    )
    
    # Create data loaders
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create optimizer (if not using DeepSpeed)
    optimizer = None
    lr_scheduler = None
    if not config['use_deepspeed']:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )
    
    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        use_deepspeed=config['use_deepspeed'],
        use_wandb=args.use_wandb,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Start training
    if rank == 0:
        print(f"\nStarting training for {args.num_epochs} epochs...")
    
    trainer.train(args.num_epochs)
    
    if rank == 0:
        print("Training completed!")
    
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()