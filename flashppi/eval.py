"""
Evaluation functions for FlashPPI.
"""
import numpy as np
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__name__)


def add_prefix_to_dict(d: dict, prefix: str) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def get_clip_metrics(seq1_features, seq2_features, logit_scale, seq1_clusters=None, seq2_clusters=None, valid_keys_tensor=None, multiplier=1_000_000):
    """
    Compute CLIP retrieval metrics (R@1/5/10, mean/median rank) on a validation batch.
    """
    metrics = {}
    N = len(seq1_features)

    if seq1_clusters is not None and seq2_clusters is not None and valid_keys_tensor is not None and len(valid_keys_tensor) > 0:
        batch_keys = seq1_clusters.unsqueeze(1) * multiplier + seq2_clusters.unsqueeze(0)
        valid_keys = valid_keys_tensor.to(batch_keys.device)
        valid_match = torch.isin(batch_keys, valid_keys)
        valid_match = valid_match | torch.eye(N, dtype=torch.bool, device=valid_match.device)
    else:
        valid_match = torch.eye(N, dtype=torch.bool)

    seq1_features = seq1_features.float()
    seq2_features = seq2_features.float()
    logit_scale = logit_scale.float()

    logits_seq1_to_seq2 = logit_scale * seq1_features @ seq2_features.t()
    logits_seq2_to_seq1 = logit_scale * seq2_features @ seq1_features.t()

    for name, (logit, valid_mask) in [
        ("seq1_to_seq2", (logits_seq1_to_seq2, valid_match)),
        ("seq2_to_seq1", (logits_seq2_to_seq1, valid_match.t())),
    ]:
        ranking = torch.argsort(logit, descending=True, dim=1)
        valid_in_ranking = torch.gather(valid_mask, 1, ranking)
        first_valid_idx = valid_in_ranking.int().argmax(dim=1)
        is_valid = torch.gather(valid_in_ranking, 1, first_valid_idx.unsqueeze(1)).squeeze(1)
        ranks = torch.where(is_valid, first_valid_idx + 1, torch.tensor(N + 1)).numpy()

        metrics[f"{name}_mean_rank"] = float(ranks.mean())
        metrics[f"{name}_median_rank"] = float(np.floor(np.median(ranks)))
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = float(valid_in_ranking[:, :k].any(dim=1).float().mean().item())

    return metrics


def evaluate(model, val_loader, accelerator, training_config, clip_loss_fn, completed_steps):
    """Evaluation loop: CLIP loss and retrieval metrics on the validation set."""

    model.eval()
    total_loss_clip = 0
    num_batches = 0

    all_clip_embed1 = []
    all_clip_embed2 = []
    all_seq1_clusters = []
    all_seq2_clusters = []
    logit_scale_val = None

    logger.info("Running evaluation...")
    progress_bar = tqdm(val_loader, disable=not accelerator.is_local_main_process, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            seq1_input_ids = batch['seq1_tokens']['input_ids'].to(accelerator.device)
            seq1_attention_mask = batch['seq1_tokens']['attention_mask'].to(accelerator.device)
            seq2_input_ids = batch['seq2_tokens']['input_ids'].to(accelerator.device)
            seq2_attention_mask = batch['seq2_tokens']['attention_mask'].to(accelerator.device)

            seq1_clusters = batch.get('seq1_cluster', None)
            seq2_clusters = batch.get('seq2_cluster', None)
            if seq1_clusters is not None:
                seq1_clusters = seq1_clusters.to(accelerator.device)
                seq2_clusters = seq2_clusters.to(accelerator.device)

            outputs = model(
                seq1_input_ids=seq1_input_ids,
                seq2_input_ids=seq2_input_ids,
                seq1_attention_mask=seq1_attention_mask,
                seq2_attention_mask=seq2_attention_mask,
            )
            clip_embed1, clip_embed2, _, logit_scale, _ = outputs[:5]

            if clip_embed1 is not None:
                loss_clip = clip_loss_fn(clip_embed1, clip_embed2, logit_scale, seq1_clusters=seq1_clusters, seq2_clusters=seq2_clusters)
                total_loss_clip += loss_clip.item()

                gathered_e1 = accelerator.gather_for_metrics(clip_embed1).cpu()
                gathered_e2 = accelerator.gather_for_metrics(clip_embed2).cpu()
                all_clip_embed1.append(gathered_e1)
                all_clip_embed2.append(gathered_e2)

                if seq1_clusters is not None:
                    all_seq1_clusters.append(accelerator.gather_for_metrics(seq1_clusters).cpu())
                    all_seq2_clusters.append(accelerator.gather_for_metrics(seq2_clusters).cpu())

            if logit_scale_val is None:
                logit_scale_val = logit_scale.mean().cpu()

            num_batches += 1

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        avg_loss_clip = total_loss_clip / num_batches if num_batches > 0 else 0.0
        metrics = {"loss_clip": avg_loss_clip}

        if all_clip_embed1:
            features1 = torch.cat(all_clip_embed1, dim=0)
            features2 = torch.cat(all_clip_embed2, dim=0)
            seq1_clusters_cat = torch.cat(all_seq1_clusters, dim=0) if all_seq1_clusters else None
            seq2_clusters_cat = torch.cat(all_seq2_clusters, dim=0) if all_seq2_clusters else None

            valid_keys_tensor = None
            multiplier = 1_000_000
            if hasattr(clip_loss_fn, 'valid_keys') and clip_loss_fn.valid_keys is not None:
                valid_keys_tensor = clip_loss_fn.valid_keys
                multiplier = getattr(clip_loss_fn, 'multiplier', 1_000_000)

            logger.info(f"Computing CLIP metrics on {len(features1)} samples...")
            clip_metrics = get_clip_metrics(
                features1, features2, logit_scale_val,
                seq1_clusters=seq1_clusters_cat,
                seq2_clusters=seq2_clusters_cat,
                valid_keys_tensor=valid_keys_tensor,
                multiplier=multiplier,
            )
            metrics.update(clip_metrics)

            logger.info(
                f"Validation | Step: {completed_steps} | "
                f"CLIP Loss: {avg_loss_clip:.4f} | "
                f"R@1: {metrics.get('seq1_to_seq2_R@1', 0):.4f} | "
                f"R@5: {metrics.get('seq1_to_seq2_R@5', 0):.4f} | "
                f"Mean Rank: {metrics.get('seq1_to_seq2_mean_rank', 0):.2f}"
            )

        if training_config.use_wandb:
            accelerator.log(add_prefix_to_dict(metrics, "eval"), step=completed_steps)

        return metrics
