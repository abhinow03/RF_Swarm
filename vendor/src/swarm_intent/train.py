"""
Training loop for the STGT swarm model.

Migrated from the notebook. Changes: explicit ``device`` argument (no global),
val/test datasets reuse the TRAIN regression stats, and paths come from
``cfg.data_dir`` instead of being hardcoded.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .dataset import SwarmDataset, collate_fn
from .model import STGTModel


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, scheduler, cfg, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for graph_seqs, labels, reg_labels in loader:
        labels, reg_labels = labels.to(device), reg_labels.to(device)
        optimizer.zero_grad()
        logits, reg_out = model(graph_seqs)
        loss = (cfg.cls_weight * F.cross_entropy(logits, labels)
                + cfg.reg_weight * F.mse_loss(reg_out, reg_labels))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for graph_seqs, labels, reg_labels in loader:
        labels, reg_labels = labels.to(device), reg_labels.to(device)
        logits, reg_out = model(graph_seqs)
        loss = (cfg.cls_weight * F.cross_entropy(logits, labels)
                + cfg.reg_weight * F.mse_loss(reg_out, reg_labels))
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def train(cfg: Config, device=None, ckpt_name: str = "best_model.pt"):
    """Load splits from cfg.data_dir, train with early stopping, save best."""
    device = device or get_device()
    d = cfg.data_dir
    load = lambda n: np.load(os.path.join(d, n))
    X_train, X_val, X_test = load("X_train.npy"), load("X_val.npy"), load("X_test.npy")
    y_train, y_val, y_test = load("y_train.npy"), load("y_val.npy"), load("y_test.npy")

    train_ds = SwarmDataset(X_train, y_train, cfg, precompute=True)
    val_ds = SwarmDataset(X_val, y_val, cfg, precompute=True,
                          reg_mean=train_ds.reg_mean, reg_std=train_ds.reg_std)
    test_ds = SwarmDataset(X_test, y_test, cfg, precompute=True,
                           reg_mean=train_ds.reg_mean, reg_std=train_ds.reg_std)

    dl = lambda ds, sh: DataLoader(ds, batch_size=cfg.batch_size, shuffle=sh,
                                   collate_fn=collate_fn, num_workers=0)
    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    model = STGTModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Linear warmup then cosine decay to a NONZERO floor (cfg.lr_min_frac * cfg.lr),
    # not OneCycleLR's near-zero final_div_factor -- see cfg.warmup_pct/lr_min_frac docstring.
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = max(1, int(cfg.warmup_pct * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.lr_min_frac + (1 - cfg.lr_min_frac) * cos_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val, patience_ctr = float("inf"), 0
    ckpt_path = os.path.join(d, ckpt_name)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, device)
        va_loss, va_acc = evaluate(model, val_loader, cfg, device)
        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)
        print(f"Epoch {epoch:3d}/{cfg.epochs} | train {tr_loss:.4f}/{tr_acc:.3f} | "
              f"val {va_loss:.4f}/{va_acc:.3f} | {time.time()-t0:.1f}s")

        if va_loss < best_val:
            best_val, patience_ctr = va_loss, 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": va_loss, "val_acc": va_acc, "cfg": cfg.to_dict(),
                "reg_mean": train_ds.reg_mean, "reg_std": train_ds.reg_std,
            }, ckpt_path)
            print(f"  saved best (val_loss={va_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, cfg, device)
    print(f"Test loss {test_loss:.4f} | test acc {test_acc:.4f} | best epoch {ckpt['epoch']}")
    return model, history, test_acc
