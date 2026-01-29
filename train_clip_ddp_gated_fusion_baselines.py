#!/usr/bin/env python3
"""
train_clip_ddp_gated_fusion_baselines.py (single script, patched for LiDAR gating + ICML table runs)

Key upgrades:
- ✅ Gate-q uses BEV *density* by default (fixes LiDAR gate saturation)
- ✅ gate_q_mode: density | logsum | density_logsum
- ✅ train_mode: cam_only | sensor_only | fusion_only | joint (true baselines)
- ✅ tune_heads applies only to heads relevant to train_mode (table-friendly)
- ✅ optional train-time image degradation augmentation for robustness story

Recommended table runs:
Run 4 modes for RADAR CSV, then repeat for LiDAR CSV.
"""

import os
import json
import random
import argparse
from collections import Counter
from typing import Optional, Tuple, Callable, Dict, Any, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import torchvision.transforms as T
import open_clip
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

# nuScenes point clouds
try:
    from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
except Exception as e:
    RadarPointCloud = None
    LidarPointCloud = None
    _NUSCENES_IMPORT_ERROR = e


# ============================================================
#  Basic DDP utilities
# ============================================================
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_dist() else 0

def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main() -> bool:
    return rank() == 0

def barrier():
    if is_dist():
        dist.barrier()

def ddp_setup(local_rank: int):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

def ddp_cleanup():
    if is_dist():
        dist.destroy_process_group()

def _to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# ============================================================
#  Image corruptions
# ============================================================
class DegradeSeverity:
    def __init__(self, severity=3):
        assert 1 <= severity <= 5
        self.s = severity

    def __call__(self, img: Image.Image) -> Image.Image:
        s = self.s
        blur_radius = {1: 0.5, 2: 1.0, 3: 2.0, 4: 3.0, 5: 5.0}[s]
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        b_rng = {1: (0.85, 1.0), 2: (0.70, 0.90), 3: (0.50, 0.80),
                 4: (0.35, 0.65), 5: (0.20, 0.50)}[s]
        c_rng = {1: (0.90, 1.0), 2: (0.75, 0.95), 3: (0.55, 0.85),
                 4: (0.40, 0.70), 5: (0.25, 0.60)}[s]
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*b_rng))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*c_rng))
        return img

class RandomDegrade:
    """Apply DegradeSeverity with probability p and random severity in [1, max_sev]."""
    def __init__(self, p: float = 0.0, max_sev: int = 3):
        self.p = float(p)
        self.max_sev = int(max_sev)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0.0:
            return img
        if random.random() > self.p:
            return img
        sev = random.randint(1, max(1, min(5, self.max_sev)))
        return DegradeSeverity(sev)(img)

class Cutout:
    def __init__(self, frac=0.35, p=1.0):
        self.frac = frac
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        W, H = img.size
        w = max(1, int(W * self.frac))
        h = max(1, int(H * self.frac))
        x0 = random.randint(0, max(0, W - w))
        y0 = random.randint(0, max(0, H - h))
        img = img.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(0, 0, 0))
        return img


# ============================================================
#  BEV helpers (vectorized + log1p + p99 norm)
# ============================================================
def points_to_bev_counts(xs, ys, x_min, x_max, y_min, y_max, H, W) -> np.ndarray:
    bev = np.zeros((H, W), dtype=np.float32)
    if xs.size == 0:
        return bev

    x_norm = (xs - x_min) / max((x_max - x_min), 1e-6)
    y_norm = (ys - y_min) / max((y_max - y_min), 1e-6)

    ix = np.clip((x_norm * (W - 1)).astype(np.int64), 0, W - 1)
    iy = np.clip(((1.0 - y_norm) * (H - 1)).astype(np.int64), 0, H - 1)

    np.add.at(bev, (iy, ix), 1.0)
    return bev

def normalize_bev(counts: np.ndarray, mode: str = "logp99", p99: float = 99.0) -> np.ndarray:
    if counts.size == 0:
        return counts.astype(np.float32)

    if mode == "max":
        m = float(counts.max()) if counts.size else 0.0
        out = counts / m if m > 0 else counts
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    bev = np.log1p(counts).astype(np.float32)
    if np.any(bev > 0):
        scale = float(np.percentile(bev[bev > 0], p99))
        scale = max(scale, 1e-6)
        bev = bev / scale
    return np.clip(bev, 0.0, 1.0).astype(np.float32)

def points_to_bev_tensor(xs, ys, x_min, x_max, y_min, y_max, H, W, norm_mode="logp99", p99=99.0) -> torch.Tensor:
    counts = points_to_bev_counts(xs, ys, x_min, x_max, y_min, y_max, H, W)
    bev = normalize_bev(counts, mode=norm_mode, p99=p99)
    return torch.from_numpy(bev[None, ...])  # (1,H,W)


# ============================================================
#  Sensor loaders (RADAR / LiDAR -> BEV tensor)
# ============================================================
def _subsample_points(xs: np.ndarray, ys: np.ndarray, drop_p: float) -> Tuple[np.ndarray, np.ndarray]:
    if drop_p <= 0.0:
        return xs, ys
    n = xs.size
    if n == 0:
        return xs, ys
    keep = max(1, int(n * (1.0 - drop_p)))
    idx = np.random.choice(n, size=keep, replace=False)
    return xs[idx], ys[idx]

def load_radar_tensor(path, x_min, x_max, y_min, y_max, H, W, norm_mode, p99, point_drop_p=0.0) -> torch.Tensor:
    if RadarPointCloud is None:
        raise ImportError(f"Could not import nuScenes RadarPointCloud. Import error: {_NUSCENES_IMPORT_ERROR}")

    try:
        pc = RadarPointCloud.from_file(path)
    except Exception as e:
        if is_main():
            print(f"[Radar] ERROR reading {path}: {e}. Returning zeros.", flush=True)
        return torch.zeros((1, H, W), dtype=torch.float32)

    pts = pc.points
    xs = pts[0, :]
    ys = pts[1, :]

    mask = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
    xs = xs[mask]
    ys = ys[mask]
    xs, ys = _subsample_points(xs, ys, point_drop_p)
    return points_to_bev_tensor(xs, ys, x_min, x_max, y_min, y_max, H, W, norm_mode=norm_mode, p99=p99)

def load_lidar_tensor(path, x_min, x_max, y_min, y_max, H, W, norm_mode, p99, z_min=-3.0, z_max=3.0, point_drop_p=0.0) -> torch.Tensor:
    if LidarPointCloud is None:
        raise ImportError(f"Could not import nuScenes LidarPointCloud. Import error: {_NUSCENES_IMPORT_ERROR}")

    try:
        pc = LidarPointCloud.from_file(path)
    except Exception as e:
        if is_main():
            print(f"[LiDAR] ERROR reading {path}: {e}. Returning zeros.", flush=True)
        return torch.zeros((1, H, W), dtype=torch.float32)

    pts = pc.points
    xs = pts[0, :]
    ys = pts[1, :]
    zs = pts[2, :]

    mask = (
        (xs >= x_min) & (xs <= x_max) &
        (ys >= y_min) & (ys <= y_max) &
        (zs >= z_min) & (zs <= z_max)
    )
    xs = xs[mask]
    ys = ys[mask]
    xs, ys = _subsample_points(xs, ys, point_drop_p)
    return points_to_bev_tensor(xs, ys, x_min, x_max, y_min, y_max, H, W, norm_mode=norm_mode, p99=p99)

def make_sensor_loader(sensor: str, cfg: Dict[str, Any]) -> Callable[[str], torch.Tensor]:
    sensor = sensor.lower().strip()
    point_drop_p = float(cfg.get("sensor_point_drop_p", 0.0))
    norm_mode = str(cfg.get("bev_norm", "logp99"))
    p99 = float(cfg.get("bev_p99", 99.0))

    if sensor == "radar":
        def _loader(p: str) -> torch.Tensor:
            return load_radar_tensor(
                p, cfg["x_min"], cfg["x_max"], cfg["y_min"], cfg["y_max"],
                cfg["bev_h"], cfg["bev_w"], norm_mode=norm_mode, p99=p99, point_drop_p=point_drop_p
            )
        return _loader

    if sensor == "lidar":
        def _loader(p: str) -> torch.Tensor:
            return load_lidar_tensor(
                p, cfg["x_min"], cfg["x_max"], cfg["y_min"], cfg["y_max"],
                cfg["bev_h"], cfg["bev_w"], norm_mode=norm_mode, p99=p99,
                z_min=float(cfg.get("z_min", -3.0)),
                z_max=float(cfg.get("z_max", 3.0)),
                point_drop_p=point_drop_p
            )
        return _loader

    raise ValueError(f"Unknown sensor '{sensor}'. Choose from: radar, lidar")


# ============================================================
#  Dataset
# ============================================================
class FrontVehicleDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_col: str,
        label_col: str,
        sensor_col: Optional[str],
        image_transform,
        sensor_loader: Optional[Callable[[str], torch.Tensor]],
        verify_paths_n: int = 0,
        sensor_drop_prob: float = 0.0,
        warn_empty_prob: float = 0.0005,
        name: str = "dataset",
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.image_col = image_col
        self.label_col = label_col
        self.sensor_col = sensor_col
        self.image_transform = image_transform
        self.sensor_loader = sensor_loader
        self.sensor_drop_prob = float(sensor_drop_prob)
        self.warn_empty_prob = float(warn_empty_prob)
        self.name = name

        missing = []
        if self.image_col not in self.df.columns:
            missing.append(self.image_col)
        if self.label_col not in self.df.columns:
            missing.append(self.label_col)
        if missing:
            raise ValueError(f"[{self.name}] Missing required columns in {csv_path}: {missing}")

        self.enable_sensor = False
        if sensor_col is not None and (sensor_col in self.df.columns) and (sensor_loader is not None):
            self.enable_sensor = True
            if is_main():
                print(f"[{self.name}] Sensor enabled from column '{sensor_col}' in {csv_path}", flush=True)
        else:
            if is_main():
                if sensor_col is None:
                    print(f"[{self.name}] Sensor disabled (sensor_col=None).", flush=True)
                elif sensor_col not in self.df.columns:
                    print(f"[{self.name}] WARNING: sensor_col '{sensor_col}' not found in {csv_path}. Running camera-only.", flush=True)
                else:
                    print(f"[{self.name}] Sensor disabled (no loader).", flush=True)

        if is_main():
            y = self.df[self.label_col].astype(int).values
            c = Counter(y.tolist())
            print(f"[{self.name}] label counts: {dict(c)}", flush=True)

        if verify_paths_n > 0 and is_main():
            self._verify_paths(n=verify_paths_n)

    def _verify_paths(self, n: int = 5):
        n = min(n, len(self.df))
        bad_img = 0
        bad_sensor = 0
        for i in range(n):
            row = self.df.iloc[i]
            img_path = str(row[self.image_col])
            if not os.path.isfile(img_path):
                bad_img += 1
                print(f"[{self.name}] WARNING: missing image_path: {img_path}", flush=True)

            if self.enable_sensor:
                s_path = str(row[self.sensor_col])
                if not os.path.isfile(s_path):
                    bad_sensor += 1
                    print(f"[{self.name}] WARNING: missing {self.sensor_col}: {s_path}", flush=True)

        print(f"[{self.name}] path check: missing images={bad_img}, missing sensor_files={bad_sensor}", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = str(row[self.image_col])
        label = int(row[self.label_col])

        img = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)

        if self.enable_sensor:
            s_path = str(row[self.sensor_col])
            sensor = self.sensor_loader(s_path)  # (1,H,W)

            if self.warn_empty_prob > 0.0:
                nnz = int((sensor > 0).sum().item())
                if nnz == 0 and random.random() < self.warn_empty_prob:
                    print(f"[WARN] Empty BEV for {s_path}", flush=True)

            if self.sensor_drop_prob > 0.0 and random.random() < self.sensor_drop_prob:
                sensor = torch.zeros_like(sensor)

            return img, sensor, label

        return img, label


# ============================================================
#  Gate-q computation (FIXES LiDAR gate saturation)
# ============================================================
def gate_q_dim(gate_q_mode: str) -> int:
    m = gate_q_mode.lower().strip()
    if m in ["density", "logsum"]:
        return 1
    if m in ["density_logsum", "density+logsum", "both"]:
        return 2
    raise ValueError(f"Unknown gate_q_mode={gate_q_mode}")

def compute_gate_q(sensor_bev: torch.Tensor, gate_q_mode: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns q: (B, qdim)
    - density: mean((bev>0)) in [0,1]
    - logsum: log1p(sum(bev))
    - density_logsum: concat([density, logsum])
    """
    m = gate_q_mode.lower().strip()
    if m == "density":
        q = (sensor_bev > 0).float().mean(dim=(1, 2, 3)).unsqueeze(-1).to(dtype)
        return q
    if m == "logsum":
        q = torch.log1p(sensor_bev.sum(dim=(1, 2, 3))).unsqueeze(-1).to(dtype)
        return q
    if m in ["density_logsum", "density+logsum", "both"]:
        q_density = (sensor_bev > 0).float().mean(dim=(1, 2, 3)).to(dtype)
        q_logsum = torch.log1p(sensor_bev.sum(dim=(1, 2, 3))).to(dtype)
        return torch.stack([q_density, q_logsum], dim=1)  # (B,2)
    raise ValueError(f"Unknown gate_q_mode={gate_q_mode}")


# ============================================================
#  Model blocks
# ============================================================
class SimpleSensorEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.proj(h)

class GatedFusion(nn.Module):
    """
    fused = v + g*s
    g = sigmoid(MLP([v,s,q]))
    """
    def __init__(self, dim: int, q_dim: int = 1, gate_hidden: int = 256):
        super().__init__()
        self.lin_v = nn.Linear(dim, dim)
        self.lin_s = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2 + q_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
        )

    def forward(self, v: torch.Tensor, s: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v2 = self.lin_v(v)
        s2 = self.lin_s(s)
        gate_in = torch.cat([v2, s2, q], dim=-1)
        g = torch.sigmoid(self.gate(gate_in))  # (B,1)
        fused = v2 + g * s2
        return fused, g

class MultiHeadBinary(nn.Module):
    def __init__(self, feat_dim: int, use_sensor: bool, sensor_in_channels: int = 1, gate_hidden: int = 256, gate_q_mode: str = "density"):
        super().__init__()
        self.use_sensor = bool(use_sensor)
        self.gate_q_mode = gate_q_mode
        self.q_dim = gate_q_dim(gate_q_mode) if self.use_sensor else 1

        self.cam_head = nn.Linear(feat_dim, 2)

        self.sensor_encoder = SimpleSensorEncoder(sensor_in_channels, feat_dim) if self.use_sensor else None
        self.sensor_head = nn.Linear(feat_dim, 2) if self.use_sensor else None

        self.fuser = GatedFusion(dim=feat_dim, q_dim=self.q_dim, gate_hidden=gate_hidden) if self.use_sensor else None
        self.fusion_head = nn.Linear(feat_dim, 2) if self.use_sensor else None

    def forward(
        self,
        vision_feat: Optional[torch.Tensor],
        sensor_bev: Optional[torch.Tensor],
        compute_cam: bool = True,
        compute_sensor: bool = True,
        compute_fusion: bool = True,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        if compute_cam:
            if vision_feat is None:
                raise ValueError("compute_cam=True requires vision_feat")
            out["cam_logits"] = self.cam_head(vision_feat)

        if (not self.use_sensor) or (sensor_bev is None):
            dev = None
            if vision_feat is not None:
                dev = vision_feat.device
                B = vision_feat.size(0)
            elif sensor_bev is not None:
                dev = sensor_bev.device
                B = sensor_bev.size(0)
            else:
                dev = torch.device("cpu")
                B = 1
            out["gate"] = torch.zeros((B, 1), device=dev)
            return out

        s_feat = self.sensor_encoder(sensor_bev)

        if compute_sensor:
            out["sensor_logits"] = self.sensor_head(s_feat)

        if compute_fusion:
            if vision_feat is None:
                raise ValueError("compute_fusion=True requires vision_feat")
            q = compute_gate_q(sensor_bev, self.gate_q_mode, dtype=vision_feat.dtype)  # (B, qdim)
            fused_feat, g = self.fuser(vision_feat, s_feat, q)
            out["fusion_logits"] = self.fusion_head(fused_feat)
            out["gate"] = g

        return out


def set_trainable_by_mode(model: nn.Module, train_mode: str, use_sensor: bool):
    for p in model.parameters():
        p.requires_grad = False

    train_mode = train_mode.lower().strip()

    if train_mode in ["cam_only", "joint"]:
        for p in model.cam_head.parameters():
            p.requires_grad = True

    if not use_sensor:
        if train_mode in ["sensor_only", "fusion_only"]:
            raise ValueError(f"train_mode={train_mode} requires sensor, but use_sensor=False")
        return

    if train_mode in ["sensor_only", "joint", "fusion_only"]:
        for p in model.sensor_encoder.parameters():
            p.requires_grad = True

    if train_mode in ["sensor_only", "joint"]:
        for p in model.sensor_head.parameters():
            p.requires_grad = True

    if train_mode in ["fusion_only", "joint"]:
        for p in model.fuser.parameters():
            p.requires_grad = True
        for p in model.fusion_head.parameters():
            p.requires_grad = True

    if train_mode not in ["cam_only", "sensor_only", "fusion_only", "joint"]:
        raise ValueError(f"Unknown train_mode={train_mode}")


# ============================================================
#  Metrics (binary)
# ============================================================
def cm_add(cm: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor):
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[int(t.item()), int(p.item())] += 1

def cm_acc(cm: torch.Tensor) -> float:
    return float(cm.diag().sum().item() / cm.sum().clamp_min(1).item())

def cm_bal_acc(cm: torch.Tensor) -> float:
    cm_f = cm.float()
    per_class = (cm_f.diag() / cm_f.sum(dim=1).clamp_min(1)).tolist()
    return float(sum(per_class) / len(per_class))

def cm_precision_recall_f1_pos(cm: torch.Tensor) -> Tuple[float, float, float]:
    tp = cm[1, 1].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    denom = 2 * tp + fp + fn
    f1 = float((2 * tp) / denom) if denom > 0 else 0.0
    return prec, rec, f1


# ============================================================
#  Which heads are meaningful for each train_mode
# ============================================================
def heads_for_mode(train_mode: str, use_sensor: bool) -> List[str]:
    tm = train_mode.lower().strip()
    if tm == "cam_only":
        return ["cam"]
    if tm == "sensor_only":
        return ["sensor"] if use_sensor else []
    if tm == "fusion_only":
        return ["fusion"] if use_sensor else []
    if tm == "joint":
        return ["cam"] + (["sensor", "fusion", "late"] if use_sensor else [])
    raise ValueError(tm)

def default_best_head_for_mode(train_mode: str, use_sensor: bool) -> str:
    tm = train_mode.lower().strip()
    if tm == "cam_only":
        return "cam"
    if tm == "sensor_only":
        return "sensor"
    if tm == "fusion_only":
        return "fusion"
    return "fusion" if use_sensor else "cam"


# ============================================================
#  Eval (per-head threshold + BEV stats)
# ============================================================
@torch.no_grad()
def eval_head_binary_distributed(
    clip_model,
    model_heads: nn.Module,
    loader,
    device,
    thr: float,
    head_key: str,
    late_alpha: float = 0.5,
    has_sensor: bool = False,
):
    model_heads.eval()
    cm = torch.zeros((2, 2), device=device, dtype=torch.long)

    gate_sum = torch.zeros((), device=device)
    gate_n = torch.zeros((), device=device)

    empty_sum = torch.zeros((), device=device)
    total_sum = torch.zeros((), device=device)
    nnz_sum = torch.zeros((), device=device)
    bev_sum_sum = torch.zeros((), device=device)

    for batch in loader:
        if has_sensor:
            x_img, x_sensor, y = batch
            x_sensor = x_sensor.to(device, non_blocking=True)

            bev_sum = x_sensor.sum(dim=(1, 2, 3))
            nnz = (x_sensor > 0).sum(dim=(1, 2, 3)).float()
            empty = (bev_sum <= 0).float()

            empty_sum += empty.sum()
            total_sum += torch.tensor(float(x_sensor.size(0)), device=device)
            nnz_sum += nnz.sum()
            bev_sum_sum += bev_sum.float().sum()
        else:
            x_img, y = batch
            x_sensor = None

        x_img = x_img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feats = clip_model.encode_image(x_img)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        out = model_heads(feats, x_sensor, compute_cam=True, compute_sensor=True, compute_fusion=True)

        if head_key == "cam":
            p1 = torch.softmax(out["cam_logits"], dim=1)[:, 1]
        elif head_key == "sensor":
            if "sensor_logits" not in out:
                continue
            p1 = torch.softmax(out["sensor_logits"], dim=1)[:, 1]
        elif head_key == "fusion":
            if "fusion_logits" not in out:
                continue
            p1 = torch.softmax(out["fusion_logits"], dim=1)[:, 1]
            g = out.get("gate", None)
            if g is not None:
                gate_sum += g.detach().sum()
                gate_n += torch.tensor(float(g.numel()), device=device)
        elif head_key == "late":
            if ("sensor_logits" not in out) or ("cam_logits" not in out):
                continue
            p_cam = torch.softmax(out["cam_logits"], dim=1)[:, 1]
            p_sen = torch.softmax(out["sensor_logits"], dim=1)[:, 1]
            p1 = late_alpha * p_cam + (1.0 - late_alpha) * p_sen
        else:
            raise ValueError(f"Unknown head_key={head_key}")

        pred = (p1 >= thr).long()
        cm_add(cm, y, pred)

    if is_dist():
        dist.all_reduce(cm, op=dist.ReduceOp.SUM)
        dist.all_reduce(gate_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(gate_n, op=dist.ReduceOp.SUM)
        dist.all_reduce(empty_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(nnz_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(bev_sum_sum, op=dist.ReduceOp.SUM)

    cm_cpu = cm.detach().cpu()
    acc = cm_acc(cm_cpu)
    bal = cm_bal_acc(cm_cpu)
    prec, rec, f1 = cm_precision_recall_f1_pos(cm_cpu)

    gate_mean = None
    if gate_n.item() > 0:
        gate_mean = float((gate_sum / gate_n.clamp_min(1)).item())

    sensor_empty_frac = None
    sensor_mean_nnz = None
    sensor_mean_bev_sum = None
    if has_sensor and total_sum.item() > 0:
        sensor_empty_frac = float((empty_sum / total_sum.clamp_min(1)).item())
        sensor_mean_nnz = float((nnz_sum / total_sum.clamp_min(1)).item())
        sensor_mean_bev_sum = float((bev_sum_sum / total_sum.clamp_min(1)).item())

    return {
        "acc": acc,
        "bal_acc": bal,
        "precision_pos": prec,
        "recall_pos": rec,
        "f1_pos": f1,
        "cm": cm_cpu.tolist(),
        "gate_mean": gate_mean,
        "sensor_empty_frac": sensor_empty_frac,
        "sensor_mean_nnz": sensor_mean_nnz,
        "sensor_mean_bev_sum": sensor_mean_bev_sum,
    }


# ============================================================
#  Threshold tuning
# ============================================================
@torch.no_grad()
def gather_val_probs_to_rank0(
    clip_model,
    model_heads: nn.Module,
    loader,
    device,
    late_alpha: float,
    has_sensor: bool,
    head_keys: List[str],
):
    model_heads.eval()
    probs_list = []
    labels_list = []

    for batch in loader:
        if has_sensor:
            x_img, x_sensor, y = batch
            x_sensor = x_sensor.to(device, non_blocking=True)
        else:
            x_img, y = batch
            x_sensor = None

        x_img = x_img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feats = clip_model.encode_image(x_img)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out = model_heads(feats, x_sensor, compute_cam=True, compute_sensor=True, compute_fusion=True)

        cols = []
        for hk in head_keys:
            if hk == "cam":
                p1 = torch.softmax(out["cam_logits"], dim=1)[:, 1]
            elif hk == "sensor":
                p1 = torch.softmax(out["sensor_logits"], dim=1)[:, 1] if "sensor_logits" in out else torch.full((x_img.size(0),), float("nan"), device=device)
            elif hk == "fusion":
                p1 = torch.softmax(out["fusion_logits"], dim=1)[:, 1] if "fusion_logits" in out else torch.full((x_img.size(0),), float("nan"), device=device)
            elif hk == "late":
                if ("sensor_logits" in out) and ("cam_logits" in out):
                    p_cam = torch.softmax(out["cam_logits"], dim=1)[:, 1]
                    p_sen = torch.softmax(out["sensor_logits"], dim=1)[:, 1]
                    p1 = late_alpha * p_cam + (1.0 - late_alpha) * p_sen
                else:
                    p1 = torch.full((x_img.size(0),), float("nan"), device=device)
            else:
                raise ValueError(hk)
            cols.append(p1)

        probs_mat = torch.stack(cols, dim=1).detach()
        probs_list.append(probs_mat)
        labels_list.append(y.detach())

    local_probs = torch.cat(probs_list, dim=0).to(torch.float32) if probs_list else torch.empty((0, len(head_keys)), device=device, dtype=torch.float32)
    local_labels = torch.cat(labels_list, dim=0).to(torch.long) if labels_list else torch.empty((0,), device=device, dtype=torch.long)

    if not is_dist():
        return local_probs.cpu(), local_labels.cpu()

    local_n = torch.tensor([local_labels.numel()], device=device, dtype=torch.long)
    sizes = [torch.zeros_like(local_n) for _ in range(world_size())]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes) if sizes else 0

    pad_n = max_n - local_labels.numel()
    if pad_n > 0:
        pad_probs = torch.full((pad_n, len(head_keys)), float("nan"), device=device, dtype=local_probs.dtype)
        local_probs = torch.cat([local_probs, pad_probs], dim=0)
        local_labels = torch.cat([local_labels, torch.zeros((pad_n,), device=device, dtype=local_labels.dtype)], dim=0)

    gathered_probs = [torch.empty((max_n, len(head_keys)), device=device, dtype=local_probs.dtype) for _ in range(world_size())]
    gathered_labels = [torch.empty((max_n,), device=device, dtype=local_labels.dtype) for _ in range(world_size())]
    dist.all_gather(gathered_probs, local_probs)
    dist.all_gather(gathered_labels, local_labels)

    if not is_main():
        return None, None

    all_probs = []
    all_labels = []
    for i in range(world_size()):
        n_i = sizes[i]
        if n_i > 0:
            all_probs.append(gathered_probs[i][:n_i].cpu())
            all_labels.append(gathered_labels[i][:n_i].cpu())

    if not all_probs:
        return torch.empty((0, len(head_keys))), torch.empty((0,), dtype=torch.long)

    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)

def tune_threshold_from_probs(probs: torch.Tensor, labels: torch.Tensor, metric: str = "f1"):
    if probs.numel() == 0:
        return 0.5, {"best_thr": 0.5, "best_score": 0.0, "f1_pos": 0.0, "bal_acc": 0.0}

    thresholds = torch.linspace(0.01, 0.99, steps=99)
    best_thr = 0.5
    best_score = -1.0
    best_cm = None

    for thr in thresholds:
        thr_f = float(thr.item())
        pred = (probs >= thr_f).long()
        cm = torch.zeros((2, 2), dtype=torch.long)
        cm_add(cm, labels.long(), pred)
        _, _, f1 = cm_precision_recall_f1_pos(cm)
        score = f1 if metric == "f1" else cm_bal_acc(cm)
        if score > best_score:
            best_score = float(score)
            best_thr = thr_f
            best_cm = cm

    _, _, f1 = cm_precision_recall_f1_pos(best_cm)
    return float(best_thr), {
        "best_thr": float(best_thr),
        "best_score": float(best_score),
        "acc": cm_acc(best_cm),
        "bal_acc": cm_bal_acc(best_cm),
        "f1_pos": float(f1),
        "cm": best_cm.tolist(),
    }


# ============================================================
#  Checkpoint save
# ============================================================
def save_ckpt(path: str, model_heads, optimizer, epoch: int, thresholds: Dict[str, float], args, best_metric: float):
    if not is_main():
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    model_obj = model_heads.module if hasattr(model_heads, "module") else model_heads
    ckpt = {
        "epoch": int(epoch),
        "heads_state": model_obj.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "best_metric": float(best_metric),
        "config": vars(args),
    }
    torch.save(ckpt, path)
    print(f"[rank 0] saved checkpoint -> {path}", flush=True)


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_pin_memory", action="store_true")

    parser.add_argument("--label_col", default="label_vehicle_in_front")
    parser.add_argument("--image_col", default="image_path")
    parser.add_argument("--use_class_weights", action="store_true")

    parser.add_argument("--sensor", choices=["radar", "lidar"], required=True)
    parser.add_argument("--sensor_col", default=None)
    parser.add_argument("--sensor_in_channels", type=int, default=1)

    parser.add_argument("--train_mode", choices=["cam_only", "sensor_only", "fusion_only", "joint"], default="joint")

    parser.add_argument("--bev_h", type=int, default=128)
    parser.add_argument("--bev_w", type=int, default=128)
    parser.add_argument("--x_min", type=float, default=0.0)
    parser.add_argument("--x_max", type=float, default=50.0)
    parser.add_argument("--y_min", type=float, default=-20.0)
    parser.add_argument("--y_max", type=float, default=20.0)
    parser.add_argument("--z_min", type=float, default=-3.0)
    parser.add_argument("--z_max", type=float, default=3.0)

    parser.add_argument("--bev_norm", choices=["logp99", "max"], default="logp99")
    parser.add_argument("--bev_p99", type=float, default=99.0)

    parser.add_argument("--sensor_drop_prob", type=float, default=0.0)
    parser.add_argument("--sensor_point_drop_p", type=float, default=0.0)

    parser.add_argument("--late_alpha", type=float, default=0.5)
    parser.add_argument("--gate_l1", type=float, default=0.0)

    # ✅ NEW: gate-q mode (DEFAULT density fixes LiDAR gate collapse)
    parser.add_argument("--gate_q_mode", choices=["density", "logsum", "density_logsum"], default="density")

    # Optional train-time image corruption augmentation
    parser.add_argument("--train_degrade_prob", type=float, default=0.0)
    parser.add_argument("--train_degrade_max_sev", type=int, default=3)

    parser.add_argument("--no_robustness", action="store_true")

    parser.add_argument("--verify_paths_n", type=int, default=8)
    parser.add_argument("--check_dataset_only", action="store_true")

    parser.add_argument("--thr_cam", type=float, default=0.5)
    parser.add_argument("--thr_sensor", type=float, default=0.5)
    parser.add_argument("--thr_fusion", type=float, default=0.5)
    parser.add_argument("--thr_late", type=float, default=0.5)

    parser.add_argument("--tune_threshold", action="store_true")
    parser.add_argument("--tune_metric", choices=["bal_acc", "f1"], default="f1")
    parser.add_argument("--tune_heads", default="cam,sensor,fusion,late")
    parser.add_argument("--tune_threshold_last_only", action="store_true")
    parser.add_argument("--fixed_threshold", type=float, default=None)

    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--save_ckpt", type=str, default=None)
    parser.add_argument("--best_metric", choices=["f1", "bal_acc"], default="f1")
    parser.add_argument("--best_head", choices=["cam", "sensor", "fusion", "late"], default=None)

    args = parser.parse_args()

    if args.sensor_col is None:
        args.sensor_col = "lidar_path" if args.sensor == "lidar" else "radar_path"

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        ddp_setup(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank())
    random.seed(args.seed + rank())
    np.random.seed(args.seed + rank())
    torch.backends.cudnn.benchmark = True

    ws = world_size()
    per_gpu_bs = max(1, args.batch_size // ws)
    persistent_workers = (not args.no_persistent_workers) and (args.num_workers > 0)
    pin_memory = False if args.no_pin_memory else bool(args.pin_memory)

    if args.best_head is None:
        args.best_head = default_best_head_for_mode(args.train_mode, use_sensor=True)

    if is_main():
        print(
            f"[Main] open_clip={open_clip.__version__} model={args.model} pretrained={args.pretrained} "
            f"sensor={args.sensor} sensor_col={args.sensor_col} train_mode={args.train_mode} "
            f"world_size={ws} per_gpu_bs={per_gpu_bs} "
            f"BEV(H,W)=({args.bev_h},{args.bev_w}) x=[{args.x_min},{args.x_max}] y=[{args.y_min},{args.y_max}] "
            f"z=[{args.z_min},{args.z_max}] bev_norm={args.bev_norm} bev_p99={args.bev_p99} "
            f"late_alpha={args.late_alpha} gate_l1={args.gate_l1} gate_q_mode={args.gate_q_mode}",
            flush=True,
        )

    barrier()
    if is_main():
        print(f"[rank {rank()}] loading CLIP pretrained='{args.pretrained}'...", flush=True)

    try:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            args.model,
            pretrained=args.pretrained,
            force_quick_gelu=True,
        )
    except TypeError:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            args.model,
            pretrained=args.pretrained,
            quick_gelu=(args.pretrained == "openai"),
        )

    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    barrier()
    if is_main():
        print(f"[rank {rank()}] CLIP ready.", flush=True)

    cfg = {
        "bev_h": int(args.bev_h),
        "bev_w": int(args.bev_w),
        "x_min": float(args.x_min),
        "x_max": float(args.x_max),
        "y_min": float(args.y_min),
        "y_max": float(args.y_max),
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
        "sensor_point_drop_p": float(args.sensor_point_drop_p),
        "bev_norm": str(args.bev_norm),
        "bev_p99": float(args.bev_p99),
    }
    sensor_loader = make_sensor_loader(args.sensor, cfg)

    # ✅ Optional train-time image degrade augmentation
    if args.train_degrade_prob > 0.0:
        train_img_tf = T.Compose([RandomDegrade(args.train_degrade_prob, args.train_degrade_max_sev), preprocess])
    else:
        train_img_tf = preprocess

    train_ds = FrontVehicleDataset(
        args.train_csv,
        image_col=args.image_col,
        label_col=args.label_col,
        sensor_col=args.sensor_col,
        image_transform=train_img_tf,
        sensor_loader=sensor_loader,
        verify_paths_n=args.verify_paths_n,
        sensor_drop_prob=float(args.sensor_drop_prob),
        name="train",
    )
    val_ds_clean = FrontVehicleDataset(
        args.val_csv,
        image_col=args.image_col,
        label_col=args.label_col,
        sensor_col=args.sensor_col,
        image_transform=preprocess,
        sensor_loader=sensor_loader,
        verify_paths_n=args.verify_paths_n,
        sensor_drop_prob=0.0,
        name="val_clean",
    )

    use_sensor = train_ds.enable_sensor and val_ds_clean.enable_sensor
    if is_main():
        print(f"[Main] use_sensor={use_sensor}", flush=True)

    if args.train_mode in ["sensor_only", "fusion_only"] and (not use_sensor):
        raise ValueError(f"train_mode={args.train_mode} requires valid sensor, but use_sensor=False")

    # Now that we know use_sensor, fix default best_head if user didn't supply it
    if args.best_head is None:
        args.best_head = default_best_head_for_mode(args.train_mode, use_sensor)

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_dist() else None
    val_sampler = DistributedSampler(val_ds_clean, shuffle=False, drop_last=False) if is_dist() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_bs,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    val_loader_clean = DataLoader(
        val_ds_clean,
        batch_size=per_gpu_bs,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    if args.check_dataset_only:
        if is_main():
            print("[Main] check_dataset_only set -> exiting after dataset checks.", flush=True)
        ddp_cleanup()
        return

    feat_dim = clip_model.visual.output_dim
    model_heads = MultiHeadBinary(
        feat_dim=feat_dim,
        use_sensor=use_sensor,
        sensor_in_channels=args.sensor_in_channels,
        gate_hidden=256,
        gate_q_mode=args.gate_q_mode,
    ).to(device)

    set_trainable_by_mode(model_heads, args.train_mode, use_sensor)

    if is_dist():
        model_heads = torch.nn.parallel.DistributedDataParallel(
            model_heads,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    trainable_params = [p for p in model_heads.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters selected (check train_mode/use_sensor).")
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    ce = nn.CrossEntropyLoss()

    thresholds: Dict[str, float] = {
        "cam": float(args.thr_cam),
        "sensor": float(args.thr_sensor),
        "fusion": float(args.thr_fusion),
        "late": float(args.thr_late),
    }

    if args.fixed_threshold is not None:
        thresholds = {k: float(args.fixed_threshold) for k in thresholds.keys()}
        tuning_enabled = False
    else:
        tuning_enabled = bool(args.tune_threshold)

    mode_heads = heads_for_mode(args.train_mode, use_sensor)
    tune_heads_req = [h.strip() for h in str(args.tune_heads).split(",") if h.strip()]
    tune_heads = [h for h in tune_heads_req if h in mode_heads]

    history = []
    robustness = {}
    tuning_summaries_by_epoch = []
    best_val_score = -1.0
    best_thresholds = dict(thresholds)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model_heads.train()

        loss_sum = torch.zeros((), device=device)
        loss_count = torch.zeros((), device=device)

        tr_empty = torch.zeros((), device=device)
        tr_total = torch.zeros((), device=device)
        tr_nnz_sum = torch.zeros((), device=device)
        tr_bev_sum_sum = torch.zeros((), device=device)

        gate_sum = torch.zeros((), device=device)
        gate_n = torch.zeros((), device=device)

        for batch in train_loader:
            if use_sensor:
                x_img, x_sensor, y = batch
                x_sensor = x_sensor.to(device, non_blocking=True)

                bev_sum = x_sensor.sum(dim=(1, 2, 3))
                nnz = (x_sensor > 0).sum(dim=(1, 2, 3)).float()
                empty = (bev_sum <= 0).float()

                tr_empty += empty.sum()
                tr_total += torch.tensor(float(x_sensor.size(0)), device=device)
                tr_nnz_sum += nnz.sum()
                tr_bev_sum_sum += bev_sum.float().sum()
            else:
                x_img, y = batch
                x_sensor = None

            x_img = x_img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            tm = args.train_mode
            compute_cam = (tm in ["cam_only", "joint"])
            compute_sensor = (use_sensor and (tm in ["sensor_only", "joint"]))
            compute_fusion = (use_sensor and (tm in ["fusion_only", "joint"]))

            feats = None
            if compute_cam or compute_fusion:
                with torch.no_grad():
                    feats = clip_model.encode_image(x_img)
                    feats = feats / feats.norm(dim=-1, keepdim=True)

            out = model_heads(feats, x_sensor, compute_cam=compute_cam, compute_sensor=compute_sensor, compute_fusion=compute_fusion)

            loss = torch.zeros((), device=device)

            if compute_cam:
                loss = loss + ce(out["cam_logits"], y)

            if compute_sensor:
                loss = loss + ce(out["sensor_logits"], y)

            if compute_fusion:
                loss = loss + ce(out["fusion_logits"], y)
                if args.gate_l1 > 0.0:
                    g = out.get("gate", None)
                    if g is not None:
                        loss = loss + float(args.gate_l1) * g.mean()
                        gate_sum += g.detach().sum()
                        gate_n += torch.tensor(float(g.numel()), device=device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum += loss.detach()
            loss_count += 1.0

        if is_dist():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(tr_empty, op=dist.ReduceOp.SUM)
            dist.all_reduce(tr_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(tr_nnz_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(tr_bev_sum_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(gate_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(gate_n, op=dist.ReduceOp.SUM)

        avg_loss = float((loss_sum / loss_count.clamp_min(1)).item())

        train_bev_stats = None
        if use_sensor and tr_total.item() > 0:
            train_bev_stats = {
                "empty_frac": float((tr_empty / tr_total.clamp_min(1)).item()),
                "mean_nnz": float((tr_nnz_sum / tr_total.clamp_min(1)).item()),
                "mean_bev_sum": float((tr_bev_sum_sum / tr_total.clamp_min(1)).item()),
            }

        gate_mean_train = None
        if gate_n.item() > 0:
            gate_mean_train = float((gate_sum / gate_n.clamp_min(1)).item())

        # ---- tune thresholds (mode-aware)
        tuning_summary_epoch = None
        if tuning_enabled and (not args.tune_threshold_last_only or epoch == args.epochs - 1):
            probs_mat_cpu, labels_cpu = gather_val_probs_to_rank0(
                clip_model=clip_model,
                model_heads=model_heads,
                loader=val_loader_clean,
                device=device,
                late_alpha=float(args.late_alpha),
                has_sensor=use_sensor,
                head_keys=mode_heads,
            )
            if is_main():
                head_to_idx = {h: i for i, h in enumerate(mode_heads)}
                tuned = {}
                for h in tune_heads:
                    col = probs_mat_cpu[:, head_to_idx[h]]
                    ok = ~torch.isnan(col)
                    thr_h, summ_h = tune_threshold_from_probs(col[ok], labels_cpu[ok], metric=args.tune_metric)
                    thresholds[h] = float(thr_h)
                    tuned[h] = summ_h

                tuning_summary_epoch = {
                    "epoch": int(epoch),
                    "tune_metric": args.tune_metric,
                    "tuned_heads": tune_heads,
                    "summaries": tuned,
                    "thresholds": dict(thresholds),
                }
                tuning_summaries_by_epoch.append(tuning_summary_epoch)
                print(f"[epoch {epoch}] tuned thresholds: " + ", ".join([f"{h}={thresholds[h]:.2f}" for h in tune_heads]), flush=True)

            if is_dist():
                pack = torch.tensor([thresholds["cam"], thresholds["sensor"], thresholds["fusion"], thresholds["late"]], device=device, dtype=torch.float32)
                dist.broadcast(pack, src=0)
                thresholds["cam"] = float(pack[0].item())
                thresholds["sensor"] = float(pack[1].item())
                thresholds["fusion"] = float(pack[2].item())
                thresholds["late"] = float(pack[3].item())

        # ---- eval (mode-aware)
        evals = {"cam": None, "sensor": None, "fusion": None, "late": None}
        for hk in mode_heads:
            evals[hk] = eval_head_binary_distributed(
                clip_model, model_heads, val_loader_clean, device,
                thr=thresholds[hk], head_key=hk,
                late_alpha=float(args.late_alpha),
                has_sensor=use_sensor
            )

        if is_main():
            print(f"\n[epoch {epoch}] train_mode={args.train_mode} train_loss={avg_loss:.4f}", flush=True)
            if train_bev_stats is not None:
                print(f"  train BEV stats: empty={train_bev_stats['empty_frac']:.3f}  mean_nnz={train_bev_stats['mean_nnz']:.1f}  mean_bev_sum={train_bev_stats['mean_bev_sum']:.2f}", flush=True)
            if gate_mean_train is not None:
                print(f"  gate_mean_train={gate_mean_train:.4f}", flush=True)

            for hk in mode_heads:
                r = evals[hk]
                extra = ""
                if hk == "fusion" and r["gate_mean"] is not None:
                    extra = f" gate_mean={r['gate_mean']}"
                print(f"  {hk.upper():5s}: acc={r['acc']:.4f} bal={r['bal_acc']:.4f} f1+={r['f1_pos']:.4f} thr={thresholds[hk]:.2f}{extra}", flush=True)

            if use_sensor and ("fusion" in mode_heads):
                r = evals["fusion"]
                print(f"  val BEV stats: empty={r['sensor_empty_frac']:.3f} mean_nnz={r['sensor_mean_nnz']:.1f} mean_bev_sum={r['sensor_mean_bev_sum']:.2f}", flush=True)

        history.append({
            "epoch": int(epoch),
            "train_loss": float(avg_loss),
            "train_bev_stats": train_bev_stats,
            "gate_mean_train": gate_mean_train,
            "thresholds": dict(thresholds),
            "val": evals,
            "tuning_summary_epoch": tuning_summary_epoch,
        })

        chosen = evals.get(args.best_head, None)
        this_score = -1.0
        if chosen is not None:
            this_score = float(chosen["f1_pos"]) if args.best_metric == "f1" else float(chosen["bal_acc"])

        if this_score > best_val_score:
            best_val_score = this_score
            best_thresholds = dict(thresholds)
            if args.save_ckpt is not None:
                save_ckpt(args.save_ckpt, model_heads, opt, epoch, best_thresholds, args, best_val_score)

    # Robustness eval only when fusion exists (useful story)
    if (not args.no_robustness) and use_sensor and ("fusion" in mode_heads) and ("cam" in mode_heads):
        if is_main():
            print("\n--- Robustness eval (VAL degraded) ---", flush=True)

        thr_cam = float(best_thresholds["cam"])
        thr_fus = float(best_thresholds["fusion"])

        for s in [1, 2, 3, 4, 5]:
            val_ds_deg = FrontVehicleDataset(
                args.val_csv,
                image_col=args.image_col,
                label_col=args.label_col,
                sensor_col=args.sensor_col,
                image_transform=T.Compose([DegradeSeverity(s), preprocess]),
                sensor_loader=sensor_loader,
                verify_paths_n=0,
                sensor_drop_prob=0.0,
                name=f"val_deg_s{s}",
            )
            deg_sampler = DistributedSampler(val_ds_deg, shuffle=False, drop_last=False) if is_dist() else None
            val_loader_deg = DataLoader(
                val_ds_deg,
                batch_size=per_gpu_bs,
                sampler=deg_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                drop_last=False,
            )

            r_cam = eval_head_binary_distributed(clip_model, model_heads, val_loader_deg, device, thr=thr_cam, head_key="cam", late_alpha=float(args.late_alpha), has_sensor=True)
            r_fus = eval_head_binary_distributed(clip_model, model_heads, val_loader_deg, device, thr=thr_fus, head_key="fusion", late_alpha=float(args.late_alpha), has_sensor=True)

            if is_main():
                print(f"severity {s}  cam_f1={r_cam['f1_pos']:.4f}  fusion_f1={r_fus['f1_pos']:.4f}  fusion_gate={r_fus['gate_mean']}", flush=True)
            robustness[f"degrade_{s}"] = {"cam": r_cam, "fusion": r_fus}

        val_ds_cut = FrontVehicleDataset(
            args.val_csv,
            image_col=args.image_col,
            label_col=args.label_col,
            sensor_col=args.sensor_col,
            image_transform=T.Compose([Cutout(frac=0.35, p=1.0), preprocess]),
            sensor_loader=sensor_loader,
            verify_paths_n=0,
            sensor_drop_prob=0.0,
            name="val_cutout",
        )
        cut_sampler = DistributedSampler(val_ds_cut, shuffle=False, drop_last=False) if is_dist() else None
        val_loader_cut = DataLoader(
            val_ds_cut,
            batch_size=per_gpu_bs,
            sampler=cut_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
        )

        r_cam = eval_head_binary_distributed(clip_model, model_heads, val_loader_cut, device, thr=thr_cam, head_key="cam", late_alpha=float(args.late_alpha), has_sensor=True)
        r_fus = eval_head_binary_distributed(clip_model, model_heads, val_loader_cut, device, thr=thr_fus, head_key="fusion", late_alpha=float(args.late_alpha), has_sensor=True)

        if is_main():
            print(f"cutout  cam_f1={r_cam['f1_pos']:.4f}  fusion_f1={r_fus['f1_pos']:.4f}", flush=True)
        robustness["cutout"] = {"cam": r_cam, "fusion": r_fus}

    if args.save_json is not None and is_main():
        obj = {
            "thresholds_best": dict(best_thresholds),
            "best_metric": float(best_val_score),
            "best_head": args.best_head,
            "mode_heads": mode_heads,
            "tune_heads": tune_heads,
            "tuning_summaries_by_epoch": tuning_summaries_by_epoch,
            "history": history,
            "robustness": robustness,
            "config": vars(args),
        }
        obj = _to_jsonable(obj)
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(obj, f, indent=2)
        print(f"[rank 0] saved run summary to {args.save_json}", flush=True)

    ddp_cleanup()

if __name__ == "__main__":
    main()
