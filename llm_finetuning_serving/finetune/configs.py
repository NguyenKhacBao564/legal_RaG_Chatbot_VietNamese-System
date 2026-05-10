"""
Different configurations for different GPU types
"""

from dataclasses import dataclass
from typing import Optional
import torch
from datetime import datetime
@dataclass
class H200OptimizedConfig:
    """Configuration optimized for H200 GPU (141GB VRAM)"""
    # Model settings
    model_name: str = "unsloth/Llama-3.1-8B-Instruct"
    max_seq_length: int = 8192 # Long context for legal documents
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = False  # H200 can handle 16-bit
    
    # LoRA settings - High rank for complex legal reasoning
    lora_r: int = 128
    lora_alpha: int = 256  # 2 * lora_r
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    
    # Training settings - Aggressive for H200
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 50
    num_train_epochs: int = 4
    max_steps: int = -1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 3407
    
    # Optimization
    optim: str = "adamw_torch"  # Full precision
    fp16: bool = False
    bf16: bool = True
    
    # Logging
    logging_steps: int = 1
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Data
    dataset_text_field: str = "text"
    packing: bool = False
    
    # Output
    output_dir: str = "./outputs"
    run_name: str = f"vietnamese-legal-llama-h200-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_method: str = "merged_16bit"

@dataclass
class RTX3090TiOptimizedConfig:
    """Safe LoRA configuration for RTX 3090 Ti 24GB VRAM."""

    model_name: str = "unsloth/Llama-3.1-8B-Instruct"
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 20
    num_train_epochs: int = 2
    max_steps: int = -1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 3407

    optim: str = "adamw_8bit"
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True

    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    dataset_text_field: str = "text"
    packing: bool = False

    output_dir: str = "./outputs"
    run_name: str = f"vietnamese-legal-llama-3090ti-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_method: str = "lora"

def get_config(gpu_type: str = "h200"):
    """Get configuration based on GPU type"""
    configs = {
        "h200": H200OptimizedConfig,
        "3090ti": RTX3090TiOptimizedConfig,
        "rtx3090ti": RTX3090TiOptimizedConfig,
        "rtx_3090_ti": RTX3090TiOptimizedConfig,
    }
    
    if gpu_type not in configs:
        raise ValueError(f"Unknown GPU type: {gpu_type}. Available: {list(configs.keys())}")
    
    return configs[gpu_type]()

def print_config_comparison():
    """Print comparison of different configurations"""
    print("="*80)
    print("🖥️  GPU CONFIGURATION COMPARISON")
    print("="*80)
    
    configs = {
        "H200 (141GB)": H200OptimizedConfig(),
        "RTX 3090 Ti (24GB)": RTX3090TiOptimizedConfig(),
    }
    
    for name, config in configs.items():
        print(f"\n📊 {name}:")
        print(f"   Max Seq Length: {config.max_seq_length}")
        print(f"   Precision: {'4-bit' if config.load_in_4bit else '16-bit'}")
        print(f"   LoRA Rank: {config.lora_r}")
        print(f"   Batch Size: {config.per_device_train_batch_size}")
        print(f"   Gradient Steps: {config.gradient_accumulation_steps}")
        print(f"   Effective Batch: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Optimizer: {config.optim}")
    
    print("="*80)

if __name__ == "__main__":
    print_config_comparison()
