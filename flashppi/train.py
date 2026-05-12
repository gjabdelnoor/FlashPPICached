"""
accelerate launch -m flashppi.train configs_train/train.yaml
"""
import argparse
import os
import sys
import yaml
import logging
import torch
import numpy as np
import datasets
import transformers
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, concatenate_datasets
from safetensors.torch import load_file
from transformers import AutoTokenizer, get_cosine_with_min_lr_schedule_with_warmup, HfArgumentParser
from tqdm.auto import tqdm

from .config import ModelConfig, TrainingConfig
from .model import FlashPPIModel
from .loss import ClipLoss, ContactLoss, GradCacheClipLoss, PPIGradientCache
from .dataset import DataCollatorForPPI, build_interaction_keys, ClusterSampler
from .eval import evaluate

logger = get_logger(__name__)

def save_config(config_dict, output_dir):
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False)

def get_grouped_params(model, weight_decay):
    # No decay, important for logit_scale
    no_decay = ["bias", "LayerNorm.weight", "logit_scale", "alpha", "beta"]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def main(config_file, run_name=None):
    # 1. Parse Configs
    parser = HfArgumentParser((ModelConfig, TrainingConfig))
    model_config, training_config = parser.parse_yaml_file(config_file)

    if isinstance(training_config.learning_rate, str):
        training_config.learning_rate = float(training_config.learning_rate)
    if run_name:
        training_config.run_name = run_name
        training_config.output_dir = os.path.join(training_config.output_dir, run_name)

    # 2. Accelerator Setup
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        log_with="wandb" if training_config.use_wandb else None,
        kwargs_handlers=[ddp_kwargs]
    )
    set_seed(training_config.seed)
    os.makedirs(training_config.output_dir, exist_ok=True)
    
    # Configure logging to show output in terminal
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Add file handler for logging to file
    if accelerator.is_main_process:
        log_file = os.path.join(training_config.output_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        logger.logger.addHandler(file_handler)
    
    # Set verbosity for datasets and transformers
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    # Log command and configuration
    string_of_command = f"{' '.join(sys.argv)}"
    logger.info("command: " + string_of_command)
    logger.info(accelerator.state, main_process_only=True)
    logger.info(f"Training/evaluation parameters {training_config}")
    logger.info(f"Model parameters {model_config}")
    
    if training_config.use_wandb:
        init_kwargs = {}
        if training_config.run_name:
            init_kwargs["wandb"] = {"name": training_config.run_name}
        accelerator.init_trackers(
            training_config.project_name,
            config={**vars(model_config), **vars(training_config)},
            init_kwargs=init_kwargs
        )

    # 3. Data Loading
    if accelerator.is_main_process:
        logger.info(f"Loading dataset from {training_config.dataset_path}")
    try:
        hf_dataset = load_dataset(training_config.dataset_path)
    except Exception as e:
        logger.error(e)
        return

    tokenizer = AutoTokenizer.from_pretrained(model_config.plm_model_name, trust_remote_code=True)
    if training_config.train_split == "all":
        print(f"Concatenating all splits: {hf_dataset.keys()}")
        train_dataset = concatenate_datasets(hf_dataset.values())
    else:
        train_dataset = hf_dataset[training_config.train_split]
        
    val_dataset = hf_dataset[training_config.val_split].shuffle(seed=training_config.seed)

    # Truncate validation dataset if max_eval_examples is set
    if training_config.max_eval_examples > 0:
        max_examples = min(training_config.max_eval_examples, len(val_dataset))
        val_dataset = val_dataset.select(range(max_examples))

    # Extract cluster columns
    c1_train = np.array(train_dataset['seq1_cluster'], dtype=np.int64)
    c2_train = np.array(train_dataset['seq2_cluster'], dtype=np.int64)
    c1_val = np.array(val_dataset['seq1_cluster'], dtype=np.int64)
    c2_val = np.array(val_dataset['seq2_cluster'], dtype=np.int64)

    # Load and concatenate DDI dataset if specified
    if training_config.ddi_dataset is not None:
        if accelerator.is_main_process:
            logger.info(f"Loading DDI dataset from {training_config.ddi_dataset}")
        ddi_hf_dataset = load_dataset(training_config.ddi_dataset)
        ddi_train = ddi_hf_dataset['train']
        c1_ddi = np.full(len(ddi_train), -1, dtype=np.int64)
        c2_ddi = np.full(len(ddi_train), -1, dtype=np.int64)

        if accelerator.is_main_process:
            logger.info(f"Concatenating DDI dataset ({len(ddi_train)} examples) with train dataset ({len(train_dataset)} examples)")
        train_dataset = concatenate_datasets([train_dataset, ddi_train])
        c1_train = np.concatenate([c1_train, c1_ddi])
        c2_train = np.concatenate([c2_train, c2_ddi])

    # ESM tokenizer adds cls/eos tokens.
    adds_special_tokens = "esm" in model_config.plm_model_name.lower()
    train_collator = DataCollatorForPPI(
        tokenizer, 
        model_config.max_len, 
        adds_special_tokens=adds_special_tokens
    )
    val_collator = DataCollatorForPPI(tokenizer, model_config.max_len, swap_augment=False, adds_special_tokens=adds_special_tokens)
        
    # Build cluster sampler if enabled
    train_sampler = None
    if training_config.use_cluster_weighted_sampling:
        if accelerator.is_main_process:
            logger.info("Building cluster sampler...")
        train_sampler = ClusterSampler(
            c1_train, c2_train,
            ddi_sampling_prob=training_config.ddi_sampling_weight,
            seed=training_config.seed
        )
        if accelerator.is_main_process:
            logger.info(f"Cluster sampling enabled: {len(train_sampler.clusters)} clusters, "
                       f"{len(train_sampler.ddi_indices)} DDI examples, "
                       f"DDI prob={training_config.ddi_sampling_weight}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        collate_fn=train_collator,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if no sampler
        num_workers=training_config.dataloader_num_workers,
        pin_memory=True
    )
    
    # Eval uses smaller batches
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.eval_batch_size,
        collate_fn=val_collator,
        shuffle=False,
        num_workers=training_config.dataloader_num_workers,
        pin_memory=True
    )
    
    # Build Global Interaction Map (for masking false negatives)
    if accelerator.is_main_process:
        logger.info("Building global interaction map from train and validation datasets...")
    valid_keys, multiplier = build_interaction_keys(c1_train, c2_train, c1_val, c2_val)

    # 4. Model & Loss & Optimizer
    model = FlashPPIModel(
        model_config,
        swap_self_negative=training_config.swap_self_negative,
        contact_random_neg_ratio=training_config.contact_random_neg_ratio,
        contact_self_neg_ratio=training_config.contact_self_neg_ratio,
        contact_hard_neg_ratio=training_config.contact_hard_neg_ratio,
        valid_keys_tensor=valid_keys,
        multiplier=multiplier,
    )
    train_loss_fn = GradCacheClipLoss(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        valid_keys_tensor=valid_keys,
        multiplier=multiplier
    )

    optimizer = torch.optim.AdamW(
        get_grouped_params(model, training_config.weight_decay),
        lr=training_config.learning_rate
    )

    num_steps = len(train_loader) * training_config.max_epochs
    lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, int(num_steps * training_config.warmup_fraction), num_steps, min_lr_rate=0.1)

    # Load from checkpoint if specified
    if training_config.load_from_checkpoint:
        model_path = os.path.join(training_config.load_from_checkpoint, "model.safetensors")
        if os.path.exists(model_path):
            state_dict = load_file(model_path)
            model.load_state_dict(state_dict, strict=False)
            if accelerator.is_main_process:
                logger.info(f"Loaded model weights from: {model_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    # Enable gradient checkpointing if specified in config
    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if accelerator.is_main_process:
            logger.info("Gradient checkpointing enabled")
    
    # Prepare
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # Load full training state (model + optimizer + scheduler) if specified
    if training_config.resume_from_checkpoint:
        accelerator.load_state(training_config.resume_from_checkpoint)
        if accelerator.is_main_process:
            logger.info(f"Resumed full training state from checkpoint: {training_config.resume_from_checkpoint}")

    # 7. Create config dict for saving (once before training)
    config_dict = {**{k: v for k, v in vars(model_config).items()}, 
                  **{k: v for k, v in vars(training_config).items()}}
    
    # 8. Initialize Loss Components
    eval_loss_fn = ClipLoss(
        local_loss=True,
        gather_with_grad=False,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        valid_keys_tensor=valid_keys,
        multiplier=multiplier
    )
    contact_loss = ContactLoss(use_focal_loss=training_config.use_focal_loss)
    
    # Conditionally initialize GradCache or standard training loss
    use_grad_cache = training_config.grad_cache_chunk_size is not None
    if use_grad_cache:
        grad_cache = PPIGradientCache(
            model=model,
            chunk_size=training_config.grad_cache_chunk_size,
            clip_loss_fn=train_loss_fn,
            contact_loss_fn=contact_loss,
            contact_weight=training_config.contact_loss_weight,
            accelerator=accelerator,
            clip_weight=training_config.clip_loss_weight
        )
    else:
        train_loss_fn = ClipLoss(
            local_loss=False,
            gather_with_grad=True,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            valid_keys_tensor=valid_keys,
            multiplier=multiplier
        )
        grad_cache = None

    # 7. Training Loop
    if use_grad_cache:
        logger.info("***** Running Training with Custom GradCache *****")
        logger.info(f"  Effective Batch: {training_config.batch_size * accelerator.num_processes}")
        logger.info(f"  Physical Chunk:  {training_config.grad_cache_chunk_size}")
    else:
        logger.info("***** Running Training with Standard Loss *****")
        logger.info(f"  Batch Size:     {training_config.batch_size * accelerator.num_processes}")

    completed_steps = 0
    
    for epoch in range(training_config.max_epochs):
        model.train()
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            if use_grad_cache:
                clip_loss_val, contact_loss_val, logit_scale_val, top1_acc = grad_cache.step(batch)
            else:
                # Standard loss computation
                seq1_ids = batch['seq1_tokens']['input_ids']
                seq1_mask = batch['seq1_tokens']['attention_mask']
                seq2_ids = batch['seq2_tokens']['input_ids']
                seq2_mask = batch['seq2_tokens']['attention_mask']
                contact_target = batch['contact_target']
                seq1_clusters = batch.get('seq1_cluster', None)
                seq2_clusters = batch.get('seq2_cluster', None)
                
                outputs = model(
                    seq1_ids, seq2_ids, seq1_mask, seq2_mask,
                    seq1_clusters, seq2_clusters
                )
                clip_embed1, clip_embed2, contact_logits, logit_scale, contact_valid_mask, seq1_clusters, seq2_clusters = outputs[:7]
                
                clip_loss, top1_acc = train_loss_fn(clip_embed1, clip_embed2, logit_scale, seq1_clusters=seq1_clusters, seq2_clusters=seq2_clusters, compute_top1_acc=True)
                clip_loss_val = clip_loss.item()

                contact_loss_tensor = contact_loss(contact_logits, contact_target, contact_valid_mask)
                contact_loss_val = contact_loss_tensor.item()
                total_loss = training_config.clip_loss_weight * clip_loss + training_config.contact_loss_weight * contact_loss_tensor
                
                logit_scale_val = logit_scale.item()
                
                # Backward pass
                accelerator.backward(total_loss)
            
            # Gradient Clipping
            grad_norm = None
            if training_config.max_grad_norm is not None:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()

            # Optimizer Step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            
            # Logging
            if completed_steps % training_config.log_interval == 0:
                log_dict = {
                    "train/loss_clip": clip_loss_val,
                    "train/loss_contact": contact_loss_val,
                    "train/loss_total": clip_loss_val + contact_loss_val,
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/logit_scale": logit_scale_val,
                    "train/retrieval_top1_acc": top1_acc
                }
                if grad_norm is not None:
                    log_dict["train/grad_norm"] = grad_norm
                accelerator.log(log_dict, step=completed_steps)
                progress_bar.set_postfix({
                    "clip": clip_loss_val, 
                    "contact": contact_loss_val,
                    "R@1": f"{top1_acc:.3f}"
                })
            
            # Saving
            if completed_steps % training_config.save_interval == 0 and accelerator.is_main_process:
                checkpoint_dir = os.path.join(training_config.output_dir, f"step_{completed_steps}")
                accelerator.save_state(checkpoint_dir)
                save_config(config_dict, checkpoint_dir)
            
            # Evaluation
            if completed_steps % training_config.eval_interval == 0:
                evaluate(model, val_loader, accelerator, training_config, eval_loss_fn, completed_steps)
                model.train()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            checkpoint_dir = os.path.join(training_config.output_dir, f"epoch_{epoch+1}")
            accelerator.save_model(model, checkpoint_dir)
            save_config(config_dict, checkpoint_dir)

    accelerator.end_training()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default="train.yaml", nargs="?")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()    
    main(args.config_file, run_name=args.run_name)