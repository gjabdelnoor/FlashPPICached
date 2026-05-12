from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the PPI-CLIP model architecture."""
    
    plm_model_name: str = field(default="facebook/esm2_t6_8M_UR50D")
    clip_embed_dim: int = field(default=256)
    contact_embed_dim: int = field(default=256)
    max_len: int = field(default=512)
    num_frozen_layers: int = field(default=0)
    gradient_checkpointing: bool = field(default=False)

@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    
    # --- Data ---
    dataset_path: str = field()
    train_split: str = field(default="train")
    val_split: str = field(default="validation")
    ddi_dataset: Optional[str] = field(default=None)
    swap_self_negative: bool = field(default=False)
    # --- Training ---
    batch_size: int = field(default=256)
    grad_cache_chunk_size: Optional[int] = field(default=None) # Set to None to disable grad_cache and use standard loss.
    eval_batch_size: int = field(default=8)
    max_epochs: int = field(default=10)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    clip_loss_weight: float = field(default=1.0)
    contact_loss_weight: float = field(default=1.0)
    contact_random_neg_ratio: float = field(default=1.0)  # Multiplier for random negatives
    contact_self_neg_ratio: float = field(default=1.0)    # Multiplier for self negatives
    contact_hard_neg_ratio: float = field(default=0.0)    # Multiplier for hard negatives (most similar by CLIP)
    use_focal_loss: bool = field(default=False)  # Use focal loss instead of standard BCE for contact prediction
    warmup_fraction: float = field(default=0.05)
    max_grad_norm: Optional[float] = field(default=1.0)
    seed: int = field(default=42)
    
    # --- Logging & Saving ---
    output_dir: str = field(default="checkpoints")
    project_name: str = field(default="flashppi")
    run_name: Optional[str] = field(default=None)
    use_wandb: bool = field(default=True)
    log_interval: int = field(default=20)
    eval_interval: int = field(default=100)
    save_interval: int = field(default=500)
    max_eval_examples: int = field(default=1000)
    dataloader_num_workers: int = field(default=4)
    use_cluster_weighted_sampling: bool = field(default=False)  # Use cluster-weighted sampling (sample cluster uniformly, then sample example from cluster)
    ddi_sampling_weight: float = field(default=0.5)  # Probability of sampling DDI examples
    resume_from_checkpoint: Optional[str] = field(default=None)  # Path to accelerate checkpoint to resume from (loads full training state)
    load_from_checkpoint: Optional[str] = field(default=None)  # Path to checkpoint to load model weights only (no optimizer/scheduler state)