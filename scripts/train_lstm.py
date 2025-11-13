from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv(x).squeeze(-1)
        return self.head(out)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        x = self.input_proj(x)
        out = self.encoder(x)
        return self.head(out[:, -1, :])


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        log_prob = nn.functional.log_softmax(logits, dim=1)
        probs = log_prob.exp()
        one_hot = nn.functional.one_hot(targets, num_classes=logits.shape[1]).float()
        pt = (probs * one_hot).sum(dim=1)
        at = self.alpha[targets]
        loss = -at * (1 - pt) ** self.gamma * pt.clamp(min=1e-8).log()
        return loss.mean()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced sequence model trainer with walk-forward evaluation.")
    parser.add_argument("--dataset", type=Path, default=Path("data/binance_features.parquet"))
    parser.add_argument("--stats", type=Path, default=Path("data/binance_feature_stats.json"))
    parser.add_argument("--model-output", type=Path, default=Path("models/dl_model.ts"))
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--models", type=str, default="gru", help="Comma list: lstm,gru,tcn,transformer")
    parser.add_argument("--walk-forward", type=int, default=3)
    parser.add_argument("--recent-start", type=str, help="YYYY-MM-DD, optional start for fine-tune slice")
    parser.add_argument("--label-mode", choices=["triple", "binary"], default="triple")
    parser.add_argument("--use-sampler", action="store_true")
    parser.add_argument("--oversample-positive", type=int, default=1)
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--neutral-weight", type=float, default=1.0, help="Multiplier for neutral class weight when label_mode=triple")
    parser.add_argument("--scheduler", choices=["none", "cosine", "onecycle"], default="none")
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn", type=int, default=256)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--min-train-share", type=float, default=0.3, help="Minimum share of history used before first validation split")
    parser.add_argument("--holdout-start", type=str, help="Exclude data on/after this date (YYYY-MM-DD) from training")
    parser.add_argument("--evaluate-holdout", action="store_true", help="After training, report metrics on holdout slice if available")
    parser.add_argument("--finetune-share", type=float, default=0.0, help="Portion of most recent data for optional fine-tune (0-1).")
    parser.add_argument("--finetune-epochs", type=int, default=3, help="Epochs for optional fine-tune stage.")
    parser.add_argument("--finetune-lr-scale", type=float, default=0.2, help="Learning rate multiplier during fine-tune.")
    parser.add_argument("--patience", type=int, default=5, help="Epoch patience for early stopping")
    parser.add_argument("--min-epochs", type=int, default=5, help="Minimum epochs to run before early stopping can trigger")
    parser.add_argument("--scenario-column", type=str, default="scenario_label", help="Column used to split regimes.")
    parser.add_argument("--min-scenario-fraction", type=float, default=0.08, help="Minimum share of samples needed for a scenario.")
    parser.add_argument("--scenario-min-samples", type=int, default=5000, help="Minimum rows required before training a scenario model.")
    parser.add_argument("--scenario-default-name", type=str, default="global", help="Name used when no scenario column is available.")
    parser.add_argument("--scenario-include", type=str, default="", help="Comma separated scenario names to train (others skip).")
    parser.add_argument("--disable-walk-forward", action="store_true", help="Skip walk-forward validation for faster experiments.")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Randomly keep this fraction of rows for quick experiments.")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Cap total training rows for fast iteration.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def _normalize_scenario_value(value: Optional[object]) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    return text or None


def _slugify(value: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    normalized = normalized.strip("_")
    return normalized or "scenario"


def _load_features(dataset_path: Path, stats_path: Path, label_mode: str, scenario_column: Optional[str] = None):
    df = pd.read_parquet(dataset_path) if dataset_path.suffix == ".parquet" else pd.read_csv(dataset_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    feature_cols = stats["feature_columns"]
    features = df[feature_cols].values.astype(np.float32)
    mean = np.array([stats["mean"].get(col, 0.0) for col in feature_cols], dtype=np.float32)
    std = np.array([stats["std"].get(col, 1.0) for col in feature_cols], dtype=np.float32)
    std[std == 0] = 1.0
    features = (features - mean) / std
    if label_mode == "binary":
        labels = (df["label"] > 0).astype(int).values
        class_to_label = {0: 0, 1: 1}
    else:
        label_map = {-1: 0, 0: 1, 1: 2}
        labels = df["label"].map(label_map).astype(int).values
        class_to_label = {0: -1, 1: 0, 2: 1}
    meta_label = df["meta_label"].values if "meta_label" in df else None
    scenario_values = None
    if scenario_column and scenario_column in df.columns:
        raw_values = df[scenario_column].to_numpy(dtype=object)
        scenario_values = np.array([_normalize_scenario_value(val) for val in raw_values], dtype=object)
    if isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index.to_numpy()
    else:
        ts_series = None
        for candidate in ("timestamp", "open_time", "date", "time"):
            if candidate in df.columns:
                ts_series = pd.to_datetime(df[candidate], errors="coerce")
                break
        if ts_series is None:
            timestamps = np.arange(len(df))
        else:
            timestamps = ts_series.to_numpy()
    return (
        features,
        labels,
        feature_cols,
        {"mean": mean.tolist(), "std": std.tolist()},
        class_to_label,
        meta_label,
        timestamps,
        scenario_values,
    )


def _build_scenario_masks(
    scenarios: Optional[np.ndarray],
    total_len: int,
    seq_len: int,
    min_fraction: float,
    min_samples: int,
    default_name: str,
):
    base_mask = np.ones(total_len, dtype=bool)
    if scenarios is None or len(scenarios) == 0:
        return [
            {"name": default_name, "value": None, "mask": base_mask, "slug": _slugify(default_name)},
        ]
    normalized = np.array([_normalize_scenario_value(val) for val in scenarios], dtype=object)
    counts = pd.Series(normalized).value_counts(dropna=False)
    entries = []
    total = max(1, total_len)
    for value, count in counts.items():
        if value is None:
            continue
        if count < max(min_samples, seq_len * 3):
            continue
        if count / total < min_fraction:
            continue
        mask = normalized == value
        entries.append({"name": value, "value": value, "mask": mask, "slug": _slugify(value)})
    has_default = any(entry["value"] is None for entry in entries)
    if not has_default:
        entries.append({"name": default_name, "value": None, "mask": base_mask, "slug": _slugify(default_name)})
    return entries


def _apply_recent_start(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: Sequence,
    recent_start: Optional[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not recent_start:
        return features, labels
    try:
        boundary = pd.Timestamp(recent_start)
    except Exception:
        return features, labels
    ts = pd.to_datetime(timestamps)
    idx = int(ts.searchsorted(boundary))
    if idx >= len(features) - seq_len - 1:
        return features, labels
    idx = max(seq_len, idx)
    return features[idx:], labels[idx:]


def _make_validation_loader(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    batch_size: int,
) -> Optional[DataLoader]:
    if len(features) <= seq_len + 1:
        return None
    val_portion = max(seq_len + batch_size, int(len(features) * 0.2))
    val_portion = min(len(features), val_portion)
    if val_portion <= seq_len:
        return None
    val_feat = features[-val_portion:]
    val_lbl = labels[-val_portion:]
    dataset = SequenceDataset(val_feat, val_lbl, seq_len)
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def _apply_sampling(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps,
    scenarios: Optional[np.ndarray],
    sample_rate: float,
    max_samples: int,
    seed: int,
    seq_len: int,
):
    if sample_rate >= 0.999 and (not max_samples or max_samples <= 0):
        return features, labels, timestamps, scenarios
    total = len(features)
    if total <= seq_len * 2:
        return features, labels, timestamps, scenarios
    rng = np.random.default_rng(seed)
    keep = total
    if sample_rate < 0.999:
        keep = max(seq_len * 2, int(total * max(sample_rate, 0.05)))
    if max_samples and max_samples > 0:
        keep = min(keep, max_samples)
    keep = min(keep, total)
    idx = rng.choice(total, size=keep, replace=False)
    idx.sort()
    features = features[idx]
    labels = labels[idx]
    timestamps = np.array(timestamps)[idx]
    if scenarios is not None:
        scenarios = scenarios[idx]
    return features, labels, timestamps, scenarios


def _log_sequence_report(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    prefix: str,
    log_fn,
):
    if len(features) <= seq_len + 1:
        return
    dataset = SequenceDataset(features, labels, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    report = _evaluate_model(model, loader, device)
    if report:
        log_fn(f"[Scenario Summary] {prefix}")
        log_fn(report)


def _build_walk_forward_splits(
    total_len: int, seq_len: int, folds: int, batch_size: int, min_share: float
) -> List[Tuple[int, int]]:
    usable = total_len - seq_len
    if usable <= 0:
        return [(seq_len, total_len)]
    min_share = max(0.05, min(0.8, min_share))
    min_train = max(int(total_len * min_share), seq_len + batch_size)
    if min_train >= total_len - seq_len:
        return [(seq_len, total_len)]
    first_val_start = max(seq_len + 1, min_train)
    if first_val_start + seq_len >= total_len:
        return [(seq_len, total_len)]
    remaining = total_len - first_val_start
    fold_size = max(seq_len * 2, remaining // max(folds, 1))
    splits = []
    val_start = first_val_start
    while val_start + seq_len < total_len:
        val_end = min(total_len, val_start + fold_size)
        splits.append((val_start, val_end))
        if len(splits) >= folds or val_end >= total_len:
            break
        val_start = val_end
    if not splits:
        splits.append((seq_len, total_len))
    return splits


def _build_scheduler(args, optimizer, steps_per_epoch: int):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs), "epoch"
    if args.scheduler == "onecycle":
        total_steps = max(1, steps_per_epoch * args.epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
        return scheduler, "batch"
    return None, None


def _prepare_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    batch_size: int,
    use_sampler: bool,
    num_classes: int,
    label_mode: str,
    oversample_positive: int,
) -> DataLoader:
    dataset = SequenceDataset(features, labels, seq_len)
    if use_sampler:
        effective_labels = dataset.labels[seq_len:]
        if len(effective_labels) > 0:
            counts = np.bincount(effective_labels, minlength=num_classes)
            weights = len(effective_labels) / (num_classes * np.maximum(counts, 1))
            sample_weights = weights[effective_labels]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
    if label_mode == "binary" and oversample_positive > 1:
        pos_idx = np.where(labels == 1)[0]
        if len(pos_idx) > 0:
            extra_idx = np.repeat(pos_idx, oversample_positive - 1)
            features = np.concatenate([features, features[extra_idx]], axis=0)
            labels = np.concatenate([labels, labels[extra_idx]], axis=0)
            dataset = SequenceDataset(features, labels, seq_len)
    drop_last = len(dataset) >= batch_size * 2
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)


def _train_one_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience,
    min_epochs,
    log_fn=print,
    scheduler_factory=None,
):
    best_val = float("inf")
    patience_left = patience
    best_state = None
    last_logits = []
    last_targets = []
    scheduler = None
    scheduler_mode = None
    steps_per_epoch = len(train_loader)
    if scheduler_factory and steps_per_epoch > 0:
        scheduler, scheduler_mode = scheduler_factory(optimizer, steps_per_epoch)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler and scheduler_mode == "batch":
                scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))

        model.eval()
        losses = []
        preds = []
        targets = []
        logits_collector = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                losses.append(loss.item())
                logits_collector.append(logits.cpu())
                preds.append(torch.argmax(logits, dim=1).cpu())
                targets.append(yb.cpu())
        val_loss = np.mean(losses) if losses else avg_loss
        if preds and targets:
            report = classification_report(
                torch.cat(targets).numpy(),
                torch.cat(preds).numpy(),
                output_dict=False,
                zero_division=0,
            )
            log_fn(report)
            last_logits = logits_collector
            last_targets = targets
        log_fn(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
        if scheduler and scheduler_mode == "epoch":
            scheduler.step()
        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
            best_state = model.state_dict()
        else:
            patience_left -= 1
            if epoch >= min_epochs and patience_left <= 0:
                log_fn("Early stopping.")
                break
    return best_state, last_logits, last_targets


def _temperature_scale(logits_list, targets_list, device):
    if not logits_list:
        return 1.0
    logits = torch.cat(logits_list).to(device)
    targets = torch.cat(targets_list).to(device)
    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().cpu().item())


def _evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu())
            targets.append(yb)
    if preds and targets:
        report = classification_report(
            torch.cat(targets).numpy(),
            torch.cat(preds).numpy(),
            zero_division=0,
            output_dict=False,
        )
        return report
    return None


def train() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    (
        features,
        labels,
        feature_cols,
        norm_stats,
        class_to_label,
        meta_label,
        timestamps,
        scenarios,
    ) = _load_features(args.dataset, args.stats, args.label_mode, scenario_column=args.scenario_column)
    orig_len = len(features)
    (
        features,
        labels,
        timestamps,
        scenarios,
    ) = _apply_sampling(
        features,
        labels,
        timestamps,
        scenarios,
        args.sample_rate,
        args.max_train_samples,
        args.seed,
        args.seq_len,
    )
    if len(features) != orig_len:
        print(
            f"Sampling dataset for fast training: kept {len(features)} / {orig_len} rows "
            f"(rate={args.sample_rate}, max={args.max_train_samples})."
        )
    device = torch.device(args.device)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    if not models:
        raise SystemExit("No models specified via --models.")
    num_classes = len(class_to_label)
    ts_series = pd.to_datetime(timestamps)
    scenario_values = scenarios
    holdout_data = None
    if args.holdout_start:
        try:
            holdout_ts = pd.Timestamp(args.holdout_start)
            train_mask = ts_series < holdout_ts
            if train_mask.sum() <= args.seq_len:
                print(
                    f"Holdout start {args.holdout_start} leaves insufficient training samples; ignoring holdout setting."
                )
            else:
                hold_mask = ~train_mask
                if hold_mask.sum() > args.seq_len:
                    holdout_data = {
                        "features": features[hold_mask],
                        "labels": labels[hold_mask],
                        "scenarios": scenario_values[hold_mask] if scenario_values is not None else None,
                    }
                features = features[train_mask]
                labels = labels[train_mask]
                ts_series = ts_series[train_mask]
                if scenario_values is not None:
                    scenario_values = scenario_values[train_mask]
                if meta_label is not None:
                    meta_label = meta_label[train_mask]
                print(
                    f"Using {len(features)} samples for training (holdout starts at {args.holdout_start}); "
                    f"holdout samples excluded: {int((~train_mask).sum())}."
                )
        except Exception as exc:
            print(f"Failed to apply holdout-start {args.holdout_start}: {exc}")
    scenario_entries = _build_scenario_masks(
        scenario_values,
        len(features),
        args.seq_len,
        args.min_scenario_fraction,
        args.scenario_min_samples,
        args.scenario_default_name,
    )
    include_list = [
        _normalize_scenario_value(item)
        for item in args.scenario_include.split(",")
        if item.strip()
    ]
    if include_list:
        filtered_entries = []
        for entry in scenario_entries:
            key = _normalize_scenario_value(entry["value"]) or _normalize_scenario_value(entry["name"])
            if key in include_list:
                filtered_entries.append(entry)
        if not filtered_entries:
            print(f"No scenario entries match include list {include_list}; keeping original set.")
        else:
            scenario_entries = filtered_entries
    flat_models = []
    scenario_metadata = []
    args.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_dir / f"train_{args.model_output.stem}_{timestamp}.log"
    log_file = log_path.open("w", encoding="utf-8")

    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Training log: {log_path}")

    def _scheduler_factory(optimizer, steps):
        return _build_scheduler(args, optimizer, steps)

    scheduler_factory = _scheduler_factory if args.scheduler != "none" else None

    try:
        for entry in scenario_entries:
            mask = entry["mask"]
            scenario_name = entry["name"]
            scenario_value = entry["value"]
            slug = entry["slug"]
            scen_features = features[mask]
            scen_labels = labels[mask]
            scen_ts = ts_series[mask]
            if len(scen_features) <= args.seq_len * 3:
                log(f"[Scenario {scenario_name}] skipped (insufficient samples: {len(scen_features)})")
                continue
            log(
                f"[Scenario {scenario_name}] samples={len(scen_features)} | "
                f"positive={int((scen_labels == 2).sum())} | negative={int((scen_labels == 0).sum())}"
            )
            if args.disable_walk_forward:
                splits = []
            else:
                splits = _build_walk_forward_splits(
                    len(scen_features), args.seq_len, args.walk_forward, args.batch_size, args.min_train_share
                )
            scenario_models_meta = []
            for model_name in models:
                log(f"=== Scenario: {scenario_name} | Model: {model_name} ===")
                if not args.disable_walk_forward:
                    for idx, (val_start, val_end) in enumerate(splits, start=1):
                        train_feat = scen_features[:val_start]
                        train_lbl = scen_labels[:val_start]
                        val_feat = scen_features[val_start:val_end]
                        val_lbl = scen_labels[val_start:val_end]
                        if len(train_feat) <= args.seq_len or len(val_feat) <= args.seq_len:
                            continue
                        train_loader = _prepare_dataloader(
                            train_feat,
                            train_lbl,
                            args.seq_len,
                            args.batch_size,
                            args.use_sampler,
                            num_classes,
                            args.label_mode,
                            args.oversample_positive,
                        )
                        val_ds = SequenceDataset(val_feat, val_lbl, args.seq_len)
                        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
                        model = _build_model(model_name, len(feature_cols), num_classes, args).to(device)
                        class_counts = np.bincount(train_lbl, minlength=num_classes)
                        total = class_counts.sum()
                        weights = total / (len(class_counts) * np.maximum(class_counts, 1))
                        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
                        criterion = (
                            FocalLoss(weight_tensor, gamma=args.focal_gamma)
                            if args.focal_loss
                            else nn.CrossEntropyLoss(weight=weight_tensor)
                        )
                        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                        best_state, _, _ = _train_one_model(
                            model,
                            train_loader,
                            val_loader,
                            criterion,
                            optimizer,
                            device,
                            args.epochs,
                            args.patience,
                            args.min_epochs,
                            log_fn=log,
                            scheduler_factory=scheduler_factory,
                        )
                        if best_state:
                            model.load_state_dict(best_state)
                        model.eval()
                        preds = []
                        targets = []
                        with torch.no_grad():
                            for xb, yb in val_loader:
                                xb = xb.to(device)
                                logits = model(xb)
                                preds.append(torch.argmax(logits, dim=1).cpu())
                                targets.append(yb)
                        if preds and targets:
                            report = classification_report(
                                torch.cat(targets).numpy(),
                                torch.cat(preds).numpy(),
                                zero_division=0,
                                output_dict=False,
                            )
                            log(f"[Scenario {scenario_name}][Walk {idx}]")
                            log(report)

                fin_features, fin_labels = _apply_recent_start(
                    scen_features, scen_labels, scen_ts, args.recent_start, args.seq_len
                )
                if len(fin_features) <= args.seq_len * 3:
                    log(f"[Scenario {scenario_name}] insufficient samples after recent_start; skipping {model_name}.")
                    continue
                final_train_loader = _prepare_dataloader(
                    fin_features,
                    fin_labels,
                    args.seq_len,
                    args.batch_size,
                    args.use_sampler,
                    num_classes,
                    args.label_mode,
                    args.oversample_positive,
                )
                val_loader = _make_validation_loader(fin_features, fin_labels, args.seq_len, args.batch_size)
                if val_loader is None:
                    log(f"[Scenario {scenario_name}] unable to build validation loader; skipping {model_name}.")
                    continue
                model = _build_model(model_name, len(feature_cols), num_classes, args).to(device)
                class_counts = np.bincount(fin_labels, minlength=num_classes)
                total = class_counts.sum()
                weights = total / (len(class_counts) * np.maximum(class_counts, 1))
                if len(class_counts) >= 3 and args.label_mode == "triple":
                    weights[1] *= args.neutral_weight
                weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
                criterion = (
                    FocalLoss(weight_tensor, gamma=args.focal_gamma)
                    if args.focal_loss
                    else nn.CrossEntropyLoss(weight=weight_tensor)
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                best_state, logits_list, targets_list = _train_one_model(
                    model,
                    final_train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    device,
                    args.epochs,
                    args.patience,
                    args.min_epochs,
                    log_fn=log,
                    scheduler_factory=scheduler_factory,
                )
                if best_state:
                    model.load_state_dict(best_state)
                temperature = 1.0
                if args.finetune_share > 0:
                    recent_len = max(
                        args.seq_len + args.batch_size,
                        int(len(scen_features) * min(0.95, max(0.0, args.finetune_share))),
                    )
                    recent_len = min(len(scen_features), recent_len)
                    if recent_len > args.seq_len + args.batch_size:
                        ft_feats = scen_features[-recent_len:]
                        ft_labels = scen_labels[-recent_len:]
                        ft_loader = _prepare_dataloader(
                            ft_feats,
                            ft_labels,
                            args.seq_len,
                            args.batch_size,
                            args.use_sampler,
                            num_classes,
                            args.label_mode,
                            args.oversample_positive,
                        )
                        ft_val_loader = _make_validation_loader(ft_feats, ft_labels, args.seq_len, args.batch_size)
                        if ft_val_loader is not None:
                            ft_lr = max(args.lr * args.finetune_lr_scale, 1e-6)
                            ft_optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=args.weight_decay)
                            finetune_epochs = max(1, args.finetune_epochs)
                            ft_patience = max(1, args.patience // 2)
                            best_state, logits_list, targets_list = _train_one_model(
                                model,
                                ft_loader,
                                ft_val_loader,
                                criterion,
                                ft_optimizer,
                                device,
                                finetune_epochs,
                                ft_patience,
                                max(1, min(finetune_epochs, args.min_epochs)),
                                log_fn=log,
                                scheduler_factory=None,
                            )
                            if best_state:
                                model.load_state_dict(best_state)

                if args.calibrate:
                    temperature = _temperature_scale(logits_list, targets_list, device)
                base_path = args.model_output.with_suffix("")
                save_path = base_path.parent / f"{base_path.stem}_{slug}_{model_name}.ts"
                model.eval()
                dummy = torch.zeros(1, args.seq_len, len(feature_cols)).to(device)
                traced = torch.jit.trace(model, dummy, strict=False, check_trace=False)
                traced.save(str(save_path))
                model_meta = {
                    "model": model_name,
                    "path": str(save_path),
                    "temperature": float(temperature),
                    "scenario": scenario_value,
                }
                scenario_models_meta.append(model_meta)
                flat_models.append(model_meta)

                _log_sequence_report(
                    model,
                    scen_features,
                    scen_labels,
                    args.seq_len,
                    args.batch_size,
                    device,
                    f"{scenario_name} | model={model_name}",
                    log,
                )

                if args.evaluate_holdout and holdout_data and holdout_data.get("features") is not None:
                    hold_features = holdout_data["features"]
                    hold_labels = holdout_data["labels"]
                    hold_mask = None
                    if holdout_data.get("scenarios") is not None and scenario_value is not None:
                        hold_mask = holdout_data["scenarios"] == scenario_value
                    if hold_mask is not None:
                        hold_features = hold_features[hold_mask]
                        hold_labels = hold_labels[hold_mask]
                    if hold_features is not None and len(hold_features) > args.seq_len:
                        hold_ds = SequenceDataset(hold_features, hold_labels, args.seq_len)
                        if len(hold_ds) > 0:
                            hold_loader = DataLoader(hold_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
                            report = _evaluate_model(model, hold_loader, device)
                            if report:
                                log(f"[Holdout Evaluation][scenario={scenario_name}] model={model_name}")
                                log(report)
                                _log_sequence_report(
                                    model,
                                    hold_features,
                                    hold_labels,
                                    args.seq_len,
                                    args.batch_size,
                                    device,
                                    f"{scenario_name} | model={model_name} | holdout",
                                    log,
                                )

            if scenario_models_meta:
                scenario_metadata.append(
                    {
                        "name": scenario_name,
                        "value": scenario_value,
                        "slug": slug,
                        "num_samples": int(len(scen_features)),
                        "models": scenario_models_meta,
                    }
                )

        if not flat_models:
            raise SystemExit("No models were successfully trained. Check scenario settings and data availability.")

        metadata = {
            "feature_columns": feature_cols,
            "normalization": norm_stats,
            "seq_len": args.seq_len,
            "label_mode": args.label_mode,
            "label_mapping": class_to_label,
            "models": flat_models,
            "scenario_column": args.scenario_column if scenario_values is not None else None,
            "scenarios": scenario_metadata,
        }
        meta_path = args.model_output.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        log(f"Saved ensemble metadata to {meta_path}")
    finally:
        log_file.close()


def _build_model(name: str, input_dim: int, num_classes: int, args: argparse.Namespace) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return LSTMClassifier(input_dim, args.hidden_dim, args.num_layers, num_classes, args.dropout)
    if name == "gru":
        return GRUClassifier(input_dim, args.hidden_dim, args.num_layers, num_classes, args.dropout)
    if name == "tcn":
        return TCNClassifier(input_dim, args.hidden_dim, num_classes)
    if name == "transformer":
        return TransformerClassifier(
            input_dim,
            args.hidden_dim,
            args.num_layers,
            args.transformer_heads,
            args.transformer_ffn,
            num_classes,
            args.dropout,
        )
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    train()
