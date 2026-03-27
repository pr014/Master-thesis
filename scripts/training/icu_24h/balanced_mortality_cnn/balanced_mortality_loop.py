"""Train/validation steps for isolated balanced mortality CNN (no shared train_loop)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: Dict[str, Any],
) -> float:
    model.train()
    training_cfg = config.get("training", {})
    clip = training_cfg.get("gradient_clip_norm")
    total_loss = 0.0
    n = 0

    for batch in loader:
        signals = batch["signal"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item() * signals.size(0)
        n += signals.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    return_arrays: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_probs = []
    all_labels = []

    for batch in loader:
        signals = batch["signal"].to(device)
        labels = batch["label"].to(device)
        logits = model(signals)
        loss = criterion(logits, labels)
        total_loss += loss.item() * signals.size(0)
        n += signals.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(n, 1)
    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    preds = (probs_np >= float(threshold)).astype(np.int64)
    acc = float((preds == labels_np).mean()) if len(labels_np) else 0.0

    auc = 0.0
    if len(np.unique(labels_np)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(labels_np, probs_np))
        except Exception:
            auc = 0.0

    # Precision/Recall/F1 + confusion counts at the selected threshold
    tp = int(((preds == 1) & (labels_np == 1)).sum())
    tn = int(((preds == 0) & (labels_np == 0)).sum())
    fp = int(((preds == 1) & (labels_np == 0)).sum())
    fn = int(((preds == 0) & (labels_np == 1)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    out = {
        "loss": avg_loss,
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "threshold": float(threshold),
    }
    if return_arrays:
        out["probs"] = probs_np
        out["labels"] = labels_np
    return out


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = config.get("training", {}).get("optimizer", {})
    name = str(opt_cfg.get("type", "Adam")).lower()
    lr = float(opt_cfg.get("lr", 5e-4))
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    betas = opt_cfg.get("betas", (0.9, 0.999))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=tuple(betas))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=tuple(betas))
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Dict[str, Any]
) -> Tuple[Optional[Any], str]:
    sched_cfg = config.get("training", {}).get("scheduler", {})
    stype = sched_cfg.get("type")
    if stype == "ReduceLROnPlateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(sched_cfg.get("factor", 0.1)),
            patience=int(sched_cfg.get("patience", 5)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )
        return sch, "plateau"
    return None, "none"
