"""
build_notebook_for_smoketest.py - Smoketest Notebook Generator

Generates bsgal_smoketest.ipynb via nbformat.v4.
Purpose: prove the pipeline produces non-zero LVIS AP on RTX 3090.
Run:  uv run python build_notebook_for_smoketest.py
"""

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def M(source):
    return new_markdown_cell(source)

def C(source):
    return new_code_cell(source)

def _build_notebook():
    """Assemble cells into a notebook v4 node."""
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    }

    # ── Title ─────────────────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    # BSGAL Smoketest — Proof that AP > 0

    This is a **minimal smoketest** notebook that trains Mask R-CNN on a small
    LVIS subset for a few epochs, then evaluates with the corrected
    (subset-matched) annotation JSON.

    **Goal:** Verify the full pipeline runs end-to-end and produces non-zero
    LVIS AP metrics.  Not for final results — use ``bsgal.ipynb`` for that.

    Target: **NVIDIA RTX 3090 (24 GB VRAM)** on vast.ai.
    Expected runtime: **~20-30 minutes** (TINY mode).
    """)
    )

    # ── 1. Environment Setup ──────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 1. Environment Setup
    """)
    )

    nb.cells.append(
        C(r"""
    import os, sys, zipfile
    from pathlib import Path

    IS_COLAB = "google.colab" in sys.modules
    RUNTIME = "colab" if IS_COLAB else "local"
    print(f"[Runtime] {RUNTIME}")

    if RUNTIME == "colab":
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        else:
            print("[OK] Drive already mounted.")
        drive_base = Path(
            os.environ.get("BSGAL_COLAB_DRIVE_BASE", "/content/drive/MyDrive/BSGAL-KELOMPOK-4")
        ).expanduser()
        if not drive_base.exists():
            raise FileNotFoundError(f"[Colab] Project folder not found: {drive_base}")
        os.environ["BSGAL_BASE_DIR"] = str(drive_base)

    # ── GPU setup ─────────────────────────────────────────────────────────────
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pip", "-q", "install",
         "lvis", "pycocotools", "opencv-python-headless", "tqdm"],
        check=False,
    )

    import torch, torchvision
    torch.backends.cudnn.enabled = True
    print(f"PyTorch: {torch.__version__}  |  TorchVision: {torchvision.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}  ({p.total_memory / 1e9:.1f} GB)")
    """)
    )

    # ── 2. Imports ────────────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 2. Imports
    """)
    )

    nb.cells.append(
        C(r"""
    import os, json, math, random, copy, time, warnings
    from dataclasses import dataclass, field
    from collections import deque, defaultdict, Counter
    from pathlib import Path
    from typing import Optional, List, Dict, Tuple, Any

    import numpy as np
    import cv2
    from PIL import Image
    from tqdm.auto import tqdm

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, Sampler

    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import roi_heads as tv_roi_heads

    from pycocotools import mask as mask_utils

    if not hasattr(np, "float"):
        np.float = float

    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_float32_matmul_precision("high")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {DEVICE}")
    """)
    )

    # ── 3. Configuration ─────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 3. Configuration (Smoketest)
    """)
    )

    nb.cells.append(
        C(r'''
    @dataclass
    class Config:
        """Smoketest configuration — small subsets, few epochs, prove AP > 0."""
        @staticmethod
        def _discover_base_dir() -> str:
            env_base = os.environ.get("BSGAL_BASE_DIR")
            if env_base:
                p = Path(env_base).expanduser().resolve()
                if (p / "datasets").exists():
                    return str(p)
            cwd = Path.cwd().resolve()
            candidates = [
                cwd, cwd.parent,
                Path("/workspace/BSGAL-KELOMPOK-4"),
                Path("/workspace"),
                Path("/content/drive/MyDrive/BSGAL-KELOMPOK-4"),
                Path("/content/BSGAL-KELOMPOK-4"),
            ]
            for p in candidates:
                if (p / "datasets" / "metadata" / "lvis_v1_train_cat_info.json").exists():
                    return str(p)
                if (p / "datasets").exists():
                    return str(p)
            return str(cwd)

        BASE_DIR: str = field(default_factory=_discover_base_dir)

        # ── Smoketest schedule ───────────────────────────────────────────────
        BASELINE_EPOCHS: int = 2          # Just enough to learn something
        TRAIN_SUBSET_SIZE: Optional[int] = 500
        VAL_SUBSET_SIZE: Optional[int] = 200
        IMS_PER_BATCH: int = 4
        IMAGE_SIZE: int = 640
        NUM_WORKERS: int = 4

        # ── Optimizer ────────────────────────────────────────────────────────
        LR: float = 1e-4
        WEIGHT_DECAY: float = 1e-4
        WARMUP_ITERS: int = 100
        CLIP_GRAD_NORM: float = 1.0
        USE_AMP: bool = True

        # ── Federated Loss ───────────────────────────────────────────────────
        USE_FED_LOSS: bool = True
        FED_LOSS_NUM_CAT: int = 50
        FED_LOSS_FREQ_WEIGHT: float = 0.5

        # ── Copy-Paste ──────────────────────────────────────────────────────
        INST_POOL_MAX_SAMPLES: int = 20
        MASK_THRESHOLD: int = 127
        SCALE_MIN_FRAC: float = 10.0 / 640.0
        SCALE_MAX_FRAC: float = 0.5
        SHAPE_JITTER: float = 0.2
        BBOX_OCCLUDED_THR: int = 10
        MASK_OCCLUDED_THR: int = 300

        # ── Misc ─────────────────────────────────────────────────────────────
        NUM_CLASSES: int = 1203
        SEED: int = 42
        LOG_EVERY: int = 25

        # ── Derived Paths ────────────────────────────────────────────────────
        @property
        def DATASETS_DIR(self): return f"{self.BASE_DIR}/datasets"
        @property
        def MODELS_DIR(self): return f"{self.BASE_DIR}/models"
        @property
        def TRAIN_IMG_DIR(self): return f"{self.DATASETS_DIR}/lvis/train2017"
        @property
        def VAL_IMG_DIR(self): return f"{self.DATASETS_DIR}/lvis/val2017"
        @property
        def TRAIN_ANN_JSON(self): return f"{self.DATASETS_DIR}/lvis/lvis_v1_train.json"
        @property
        def VAL_ANN_JSON(self): return f"{self.DATASETS_DIR}/lvis/lvis_v1_val.json"
        @property
        def INST_POOL_JSON(self): return f"{self.DATASETS_DIR}/instance/output/LVIS_instance_pools.json"
        @property
        def INST_POOL_ROOT(self): return f"{self.DATASETS_DIR}/instance"
        @property
        def AREA_STATS_JSON(self): return f"{self.DATASETS_DIR}/metadata/area_mean_std2.json"
        @property
        def CAT_INFO_JSON(self): return f"{self.DATASETS_DIR}/metadata/lvis_v1_train_cat_info.json"
        @property
        def CKPT_DIR(self): return f"{self.MODELS_DIR}/smoketest"

    cfg = Config()
    Path(cfg.CKPT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"BASE_DIR:   {cfg.BASE_DIR}")
    print(f"SUBSET:     train={cfg.TRAIN_SUBSET_SIZE}  val={cfg.VAL_SUBSET_SIZE}")
    print(f"EPOCHS:     {cfg.BASELINE_EPOCHS}")
    print(f"BATCH:      {cfg.IMS_PER_BATCH}")
    ''')
    )

    # ── 4. Utilities ─────────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 4. Utility Functions
    """)
    )

    nb.cells.append(
        C(r'''
    def set_seed(seed: int):
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    set_seed(cfg.SEED)


    def cosine_warmup_lr(step, base_lr, warmup, total):
        if step < warmup:
            return base_lr * (step + 1) / max(1, warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))

    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    class MetricLogger:
        def __init__(self, window=50):
            self.data = defaultdict(lambda: deque(maxlen=window))
        def update(self, **kwargs):
            for k, v in kwargs.items():
                if v is None: continue
                if isinstance(v, torch.Tensor): v = v.item()
                self.data[k].append(float(v))
        def avg(self, key):
            d = self.data[key]
            return sum(d) / len(d) if d else 0.0
        def fmt(self, keys=None):
            keys = keys or list(self.data.keys())
            return " | ".join(f"{k}={self.avg(k):.4f}" for k in keys if self.data[k])
    ''')
    )

    # ── 5. Data Loading ──────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 5. LVIS Data Loading
    """)
    )

    nb.cells.append(
        C(r'''
    def load_cat_info(path, num_classes=1203, freq_weight_pow=0.5):
        with open(path) as f:
            items = json.load(f)
        items_sorted = sorted(items, key=lambda x: x["id"])
        assert len(items_sorted) == num_classes
        image_counts = np.array([c["image_count"] for c in items_sorted], dtype=np.float32)
        return {
            "cat_freq_weight": torch.from_numpy(image_counts ** freq_weight_pow),
            "cat_freq_class": {i: items_sorted[i]["frequency"] for i in range(num_classes)},
            "cat_names": [items_sorted[i]["name"] for i in range(num_classes)],
        }

    def load_lvis_to_records(json_path, image_dir, subset_size=None, seed=42,
                              require_annotations=True):
        print(f"[LVIS] loading {json_path} ...")
        with open(json_path) as f:
            lvis = json.load(f)
        img_index = {img["id"]: img for img in lvis["images"]}
        anns_by_img = defaultdict(list)
        skipped = 0
        for ann in lvis["annotations"]:
            valid_segs = [s for s in ann.get("segmentation", []) if len(s) >= 6]
            if not valid_segs: skipped += 1; continue
            anns_by_img[ann["image_id"]].append({
                "bbox_xywh": ann["bbox"],
                "category_id_contig": ann["category_id"] - 1,
                "segmentation": valid_segs,
                "iscrowd": ann.get("iscrowd", 0),
            })
        def _fname(img):
            fn = img.get("file_name", img.get("coco_url", f"{int(img['id']):012d}.jpg"))
            return fn.split("/")[-1]
        records = []
        for img_id, img in img_index.items():
            anns = anns_by_img.get(img_id, [])
            if require_annotations and not anns: continue
            full_path = os.path.join(image_dir, _fname(img))
            if not os.path.exists(full_path): continue
            records.append({
                "file_name": full_path, "image_id": img_id,
                "height": img["height"], "width": img["width"],
                "annotations": anns,
                "neg_category_ids_contig": [c - 1 for c in img.get("neg_category_ids", [])],
                "not_exhaustive_category_ids_contig": [c - 1 for c in img.get("not_exhaustive_category_ids", [])],
            })
        print(f"[LVIS] {len(records)} images ({skipped} bad polys skipped)")
        if subset_size is not None and subset_size < len(records):
            rng = random.Random(seed); rng.shuffle(records)
            records = records[:subset_size]
            print(f"[LVIS] subset to {len(records)} records")
        return records

    cat_info = load_cat_info(cfg.CAT_INFO_JSON)
    print(f"cat_names[:3]: {cat_info['cat_names'][:3]}")
    print(f"r/c/f counts: {Counter(cat_info['cat_freq_class'].values())}")
    ''')
    )

    # ── 6. Augmentation ──────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 6. Image & Mask Augmentation
    """)
    )

    nb.cells.append(
        C(r'''
    def polygons_to_bitmask(polygons, h, w):
        if not polygons: return np.zeros((h, w), dtype=np.uint8)
        rles = mask_utils.frPyObjects(polygons, h, w)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
        return mask_utils.decode(rle).astype(np.uint8)

    def letterbox(image, masks, size, pad_value=114):
        h, w = image.shape[:2]
        scale = size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        image_r = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        out = np.full((size, size, image.shape[2]), pad_value, dtype=image.dtype)
        out[:new_h, :new_w] = image_r
        masks_out = None
        if masks is not None and len(masks) > 0:
            masks_r = np.stack([cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                                for m in masks], axis=0)
            masks_out = np.zeros((masks_r.shape[0], size, size), dtype=np.uint8)
            masks_out[:, :new_h, :new_w] = masks_r
        elif masks is not None:
            masks_out = np.zeros((0, size, size), dtype=np.uint8)
        return out, masks_out, scale, (size - new_w, size - new_h)

    def random_hflip(image, masks, p=0.5):
        if random.random() < p:
            image = image[:, ::-1].copy()
            if masks is not None and len(masks) > 0:
                masks = masks[:, :, ::-1].copy()
        return image, masks

    def bboxes_from_masks(masks):
        if len(masks) == 0: return np.zeros((0, 4), dtype=np.float32)
        boxes = np.zeros((len(masks), 4), dtype=np.float32)
        for i, m in enumerate(masks):
            ys, xs = np.where(m > 0)
            if len(xs) == 0: continue
            boxes[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        return boxes

    def get_largest_connected_component(mask):
        if mask.sum() == 0: return mask
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if n <= 1: return mask
        return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    ''')
    )

    # ── 7. Instance Pool ─────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 7. Instance Pool (for collator compatibility)
    """)
    )

    nb.cells.append(
        C(r'''
    class InstancePool:
        def __init__(self, pool_json, root_dir, area_stats_json,
                     scale_min_frac, scale_max_frac, mask_threshold,
                     shape_jitter, max_samples):
            self.root_dir = root_dir
            self.scale_min_frac = scale_min_frac
            self.scale_max_frac = scale_max_frac
            self.mask_threshold = mask_threshold
            self.shape_jitter = shape_jitter
            self.max_samples = max_samples
            with open(pool_json) as f: raw = json.load(f)
            self.per_cat_pool = {}
            for k, v in raw.items():
                self.per_cat_pool[int(k)] = [os.path.join(root_dir, p[1:] if p.startswith("*") else p) for p in v]
            self.cats_with_pool = [c for c, p in self.per_cat_pool.items() if len(p) > 0]
            with open(area_stats_json) as f: raw_area = json.load(f)
            self.area_stats = {int(k) - 1: (float(v[0]), float(v[1])) for k, v in raw_area.items()}

        def sample_cats(self, k):
            return [random.choice(self.cats_with_pool) for _ in range(k)]

        def load_rgba_instance(self, cat_contig, base_h, base_w):
            paths = self.per_cat_pool.get(cat_contig, [])
            if not paths: return None
            try: img = np.array(Image.open(random.choice(paths)).convert("RGBA"))
            except Exception: return None
            if img.shape[0] < 4 or img.shape[1] < 4: return None
            a_mean, a_std = self.area_stats.get(cat_contig, (0.05, 0.05))
            a_std = max(a_std, 1e-3)
            area_frac = float(np.clip(a_mean + np.random.randn() * a_std,
                                       self.scale_min_frac, self.scale_max_frac))
            area_pixels = area_frac * base_h * base_w
            if area_pixels < 16: return None
            h_src, w_src = img.shape[:2]
            ratio = (w_src / h_src) * np.random.uniform(1 - self.shape_jitter, 1 + self.shape_jitter)
            target_w = max(2, int(round(math.sqrt(ratio * area_pixels))))
            target_h = max(2, int(round(target_w / max(1e-3, ratio))))
            if target_w >= base_w or target_h >= base_h:
                scale = min((base_w - 1) / target_w, (base_h - 1) / target_h) * 0.95
                target_w, target_h = max(2, int(target_w * scale)), max(2, int(target_h * scale))
            img_r = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = (img_r[..., 3] >= self.mask_threshold).astype(np.uint8)
            if mask.sum() < 16: return None
            mask = get_largest_connected_component(mask)
            if random.random() < 0.5: img_r, mask = img_r[:, ::-1].copy(), mask[:, ::-1].copy()
            img_r[..., 3] = mask * 255
            return {"rgba": img_r, "mask": mask, "label_contig": cat_contig}


    def _bbox_from_single(m):
        ys, xs = np.where(m > 0)
        if len(xs) == 0: return np.zeros(4, dtype=np.float32)
        return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)

    def paste_instances_into(image, masks, labels, instance_source, src_instances,
                              bbox_occluded_thr, mask_occluded_thr):
        H, W = image.shape[:2]
        image, masks = image.copy(), masks.copy() if len(masks) else masks
        for src in src_instances:
            rgba, smask = src["rgba"], src["mask"]
            sh, sw = smask.shape
            if sh >= H or sw >= W: continue
            x, y = random.randint(0, W - sw - 1), random.randint(0, H - sh - 1)
            alpha = smask.astype(np.float32)[..., None]
            roi = image[y:y+sh, x:x+sw].astype(np.float32)
            image[y:y+sh, x:x+sw] = np.clip(roi * (1 - alpha) + rgba[..., :3].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            paste_mask = np.zeros((H, W), dtype=np.uint8)
            paste_mask[y:y+sh, x:x+sw] = smask
            if len(masks) > 0:
                new_m, new_l, new_s = [], [], []
                for i in range(len(masks)):
                    m2 = (masks[i] & (1 - paste_mask)).astype(np.uint8)
                    if m2.sum() < 16: continue
                    new_m.append(m2); new_l.append(labels[i]); new_s.append(instance_source[i])
                masks = np.stack(new_m, axis=0) if new_m else np.zeros((0, H, W), np.uint8)
                labels = np.array(new_l, dtype=np.int64) if new_l else np.zeros((0,), np.int64)
                instance_source = np.array(new_s, dtype=np.uint8) if new_s else np.zeros((0,), np.uint8)
            masks = np.concatenate([masks, paste_mask[None]], axis=0) if len(masks) else paste_mask[None]
            labels = np.concatenate([labels, np.array([src["label_contig"]], dtype=np.int64)])
            instance_source = np.concatenate([instance_source, np.array([1], dtype=np.uint8)])
        return image, masks, labels, instance_source
    ''')
    )

    # ── 8. Dataset ───────────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 8. LVIS Mask Dataset
    """)
    )

    nb.cells.append(
        C(r'''
    class LVISMaskDataset(Dataset):
        def __init__(self, records, image_size):
            self.records = records
            self.image_size = image_size
        def __len__(self): return len(self.records)
        def __getitem__(self, idx):
            rec = self.records[idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img_bgr = cv2.imread(rec["file_name"], cv2.IMREAD_COLOR)
            if img_bgr is None:
                img_bgr = np.zeros((rec["height"], rec["width"], 3), dtype=np.uint8)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            anns = rec["annotations"]
            if anns:
                masks = np.stack([polygons_to_bitmask(a["segmentation"], h, w) for a in anns], axis=0)
                labels = np.array([a["category_id_contig"] for a in anns], dtype=np.int64)
            else:
                masks = np.zeros((0, h, w), dtype=np.uint8)
                labels = np.zeros((0,), dtype=np.int64)
            img_rgb, masks, _, _ = letterbox(img_rgb, masks, self.image_size)
            img_rgb, masks = random_hflip(img_rgb, masks)
            return {
                "image_rgb_uint8": img_rgb, "masks_uint8": masks, "labels_contig": labels,
                "instance_source": np.zeros((len(masks),), dtype=np.uint8),
                "image_id": rec["image_id"],
                "neg_category_ids_contig": rec.get("neg_category_ids_contig", []),
                "not_exhaustive_category_ids_contig": rec.get("not_exhaustive_category_ids_contig", []),
                "orig_size": (rec["height"], rec["width"]),
            }
    ''')
    )

    # ── 9. Collator ──────────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 9. Collator
    """)
    )

    nb.cells.append(
        C(r'''
    def _to_torchvision_target(sample):
        img_t = torch.from_numpy(sample["image_rgb_uint8"].transpose(2, 0, 1).copy()).float() / 255.0
        masks, labels_contig = sample["masks_uint8"], sample["labels_contig"]
        if len(masks) > 0:
            boxes = bboxes_from_masks(masks)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, masks, labels_contig = boxes[keep], masks[keep], labels_contig[keep]
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
        labels_tv = torch.from_numpy(labels_contig.astype(np.int64) + 1)
        orig_h, orig_w = sample["orig_size"]
        target = {
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "labels": labels_tv,
            "masks": torch.from_numpy(masks.astype(np.uint8)),
            "image_id": torch.tensor([sample["image_id"]]),
            "orig_size": torch.tensor([int(orig_h), int(orig_w)], dtype=torch.long),
        }
        return img_t, target


    class SimpleCollator:
        """Baseline-only collator (no copy-paste, no triplet)."""
        def __call__(self, samples):
            return [_to_torchvision_target(s) for s in samples]
    ''')
    )

    # ── 10. Sampler ──────────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 10. Repeat-Factor Sampler
    """)
    )

    nb.cells.append(
        C(r'''
    def compute_repeat_factors(records, num_classes, threshold=0.001):
        cat_image_count = np.zeros(num_classes, dtype=np.int64)
        for r in records:
            for c in {a["category_id_contig"] for a in r["annotations"]}:
                cat_image_count[c] += 1
        freqs = cat_image_count / max(1, len(records))
        rc = np.where(freqs > 0, np.maximum(1.0, np.sqrt(threshold / np.maximum(freqs, 1e-9))), 1.0)
        image_rep = np.ones(len(records), dtype=np.float32)
        for i, r in enumerate(records):
            cats = {a["category_id_contig"] for a in r["annotations"]}
            if cats: image_rep[i] = max(rc[c] for c in cats)
        return image_rep

    class RepeatFactorSampler(Sampler):
        def __init__(self, repeat_factors, seed=42):
            self.repeat_factors = repeat_factors
            self.seed = seed
        def _epoch_indices(self, epoch):
            rng = np.random.RandomState(self.seed + epoch)
            rf = self.repeat_factors
            rep_int = np.floor(rf).astype(np.int64) + (rng.rand(len(rf)) < (rf - np.floor(rf))).astype(np.int64)
            indices = []
            for i, r in enumerate(rep_int): indices.extend([i] * int(r))
            rng.shuffle(indices)
            return indices
        def __iter__(self):
            epoch = 0
            while True:
                yield from self._epoch_indices(epoch); epoch += 1
        def __len__(self): return int(self.repeat_factors.sum())

    class _FiniteFromInfiniteSampler(Sampler):
        def __init__(self, infinite_sampler, num_samples):
            self.it = infinite_sampler
            self.num_samples = num_samples
        def __iter__(self):
            gen = iter(self.it)
            for _ in range(self.num_samples): yield next(gen)
        def __len__(self): return self.num_samples
    ''')
    )

    # ── 11. Federated Loss ───────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 11. Federated Loss
    """)
    )

    nb.cells.append(
        C(r'''
    def get_fed_loss_inds(gt_classes_contig, num_sample_cats, num_classes_no_bg, freq_weight):
        appeared = torch.unique(gt_classes_contig)
        appeared = appeared[(appeared >= 0) & (appeared < num_classes_no_bg)]
        if len(appeared) >= num_sample_cats: return appeared
        prob = freq_weight.to(appeared.device).clone()
        prob[appeared] = 0.0
        if prob.sum() <= 0: return appeared
        n_extra = min(num_sample_cats - len(appeared), int((prob > 0).sum().item()))
        if n_extra <= 0: return appeared
        return torch.cat([appeared, torch.multinomial(prob, n_extra, replacement=False)])

    def federated_sigmoid_ce_loss(logits, labels_tv, freq_weight, num_sample_cats, num_classes_no_bg):
        target = torch.zeros_like(logits)
        fg_mask = labels_tv > 0
        if fg_mask.any(): target[fg_mask, labels_tv[fg_mask]] = 1.0
        gt_contig = labels_tv[fg_mask] - 1 if fg_mask.any() else torch.empty(0, dtype=torch.long, device=logits.device)
        inds = get_fed_loss_inds(gt_contig, num_sample_cats, num_classes_no_bg, freq_weight)
        inds_tv = torch.cat([torch.zeros(1, dtype=torch.long, device=logits.device), inds.to(logits.device).long() + 1])
        return F.binary_cross_entropy_with_logits(logits[:, inds_tv], target[:, inds_tv], reduction="mean")

    def build_fastrcnn_loss_patch(freq_weight, num_sample_cats, num_classes_no_bg):
        def patched(class_logits, box_regression, labels, regression_targets):
            labels_t = torch.cat(labels, dim=0)
            reg_t = torch.cat(regression_targets, dim=0)
            cls_loss = federated_sigmoid_ce_loss(class_logits, labels_t, freq_weight.to(class_logits.device),
                                                  num_sample_cats, num_classes_no_bg)
            pos = torch.where(labels_t > 0)[0]
            N = class_logits.shape[0]
            box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
            box_loss = F.smooth_l1_loss(box_regression[pos, labels_t[pos]], reg_t[pos],
                                         reduction="sum", beta=1.0/9) / max(1, labels_t.numel())
            return cls_loss, box_loss
        return patched
    ''')
    )

    # ── 12. Model ────────────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 12. Model Building
    """)
    )

    nb.cells.append(
        C(r'''
    def build_model(num_classes_with_bg, freq_weight, fed_loss_num_cat, use_fed_loss, image_size):
        model = maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
            num_classes=num_classes_with_bg,
            min_size=image_size, max_size=image_size,
            rpn_pre_nms_top_n_train=2000, rpn_post_nms_top_n_train=1000,
            rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=500,
            box_detections_per_img=300, box_score_thresh=0.0,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes_with_bg)
        if use_fed_loss:
            num_no_bg = num_classes_with_bg - 1
            tv_roi_heads.fastrcnn_loss = build_fastrcnn_loss_patch(freq_weight, fed_loss_num_cat, num_no_bg)
        return model
    ''')
    )

    # ── 13. Training Steps ───────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 13. Training & Evaluation
    """)
    )

    nb.cells.append(
        C(r'''
    def split_batch_to_device(batch_pairs, device):
        images, targets = [], []
        for img, tgt in batch_pairs:
            images.append(img.to(device, non_blocking=True))
            targets.append({k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                            for k, v in tgt.items()})
        return images, targets

    def train_step(model, optimizer, scaler, batch_pairs, device, use_amp, clip_grad_norm):
        images, targets = split_batch_to_device(batch_pairs, device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer); scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()} | {"loss_total": loss.item()}


    def predictions_to_lvis_results(model, eval_loader, device, max_dets=300):
        model.eval()
        results = []
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="[eval]", leave=False):
                images = [img.to(device) for img, _ in batch]
                targets_list = [tgt for _, tgt in batch]
                preds = model(images)
                for pred, tgt in zip(preds, targets_list):
                    image_id = int(tgt["image_id"].item())
                    boxes, scores, labels, masks = (pred[k].cpu().numpy() for k in ["boxes", "scores", "labels", "masks"])
                    if len(scores) > max_dets:
                        keep = np.argsort(-scores)[:max_dets]
                        boxes, scores, labels, masks = boxes[keep], scores[keep], labels[keep], masks[keep]
                    for i in range(len(scores)):
                        m = (masks[i, 0] >= 0.3).astype(np.uint8)
                        if not np.isfinite(m).all(): continue
                        rle = mask_utils.encode(np.asfortranarray(m))
                        if isinstance(rle["counts"], bytes): rle["counts"] = rle["counts"].decode("ascii")
                        x1, y1, x2, y2 = boxes[i]
                        results.append({
                            "image_id": image_id, "category_id": int(labels[i]),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(scores[i]), "segmentation": rle,
                        })
        return results

    def evaluate_on_lvis(model, eval_loader, ann_json, device):
        try: from lvis import LVIS, LVISResults, LVISEval
        except ImportError: print("[WARN] lvis not installed"); return {}
        preds = predictions_to_lvis_results(model, eval_loader, device)
        if not preds: return {k: 0.0 for k in ["AP", "AP50", "AP75", "APr", "APc", "APf"]}
        lvis_gt = LVIS(ann_json)
        lvis_dt = LVISResults(lvis_gt, preds)
        ev = LVISEval(lvis_gt, lvis_dt, iou_type="segm")
        ev.run()
        return {k: float(ev.results.get(k, 0.0)) for k in ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"]}
    ''')
    )

    # ── 14. DataLoader ───────────────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 14. DataLoader Construction
    """)
    )

    nb.cells.append(
        C(r"""
    def make_baseline_loader(dataset, sampler_train, iters_per_epoch):
        return DataLoader(
            dataset, batch_size=cfg.IMS_PER_BATCH,
            sampler=_FiniteFromInfiniteSampler(sampler_train, iters_per_epoch * cfg.IMS_PER_BATCH * 1000),
            num_workers=cfg.NUM_WORKERS, collate_fn=SimpleCollator(),
            pin_memory=True, persistent_workers=cfg.NUM_WORKERS > 0,
        )

    def make_eval_loader(dataset):
        return DataLoader(dataset, batch_size=1, shuffle=False,
                          num_workers=cfg.NUM_WORKERS, collate_fn=SimpleCollator(),
                          pin_memory=True, persistent_workers=cfg.NUM_WORKERS > 0)
    """)
    )

    # ── 15. Setup, Train, Eval ───────────────────────────────────────────────
    nb.cells.append(
        M(r"""
    ## 15. Setup + Baseline Training + Evaluation
    """)
    )

    nb.cells.append(
        C(r'''
    # ── Load data ─────────────────────────────────────────────────────────────
    print("[Setup] loading LVIS train records ...")
    train_records = load_lvis_to_records(cfg.TRAIN_ANN_JSON, cfg.TRAIN_IMG_DIR,
                                         subset_size=cfg.TRAIN_SUBSET_SIZE, seed=cfg.SEED)
    print("[Setup] loading LVIS val records ...")
    val_records = load_lvis_to_records(cfg.VAL_ANN_JSON, cfg.VAL_IMG_DIR,
                                       subset_size=cfg.VAL_SUBSET_SIZE, seed=cfg.SEED,
                                       require_annotations=False)

    # ── CRITICAL: Create subset annotation JSON for evaluation ────────────────
    # The LVIS evaluator computes AP over ALL images in the annotation file.
    # If we evaluate a 200-image subset against the full 20K-image val JSON,
    # images without predictions get zero recall → metrics collapse to 0.
    # Fix: filter the annotation JSON to match the eval subset exactly.
    val_image_ids = {r["image_id"] for r in val_records}
    val_ann_subset_path = os.path.join(cfg.DATASETS_DIR, "lvis", "lvis_v1_val_subset.json")
    if not os.path.exists(val_ann_subset_path):
        print(f"[Setup] Creating val subset annotation JSON ({len(val_image_ids)} images) ...")
        with open(cfg.VAL_ANN_JSON) as f:
            full_val_ann = json.load(f)
        val_ann_subset = {
            "images": [img for img in full_val_ann["images"] if img["id"] in val_image_ids],
            "annotations": [ann for ann in full_val_ann["annotations"] if ann["image_id"] in val_image_ids],
            "categories": full_val_ann["categories"],
        }
        with open(val_ann_subset_path, "w") as f:
            json.dump(val_ann_subset, f)
        print(f"[Setup] Saved: {len(val_ann_subset['images'])} imgs, {len(val_ann_subset['annotations'])} anns")
    VAL_ANN_SUBSET = val_ann_subset_path
    print(f"[Setup] Eval annotations: {VAL_ANN_SUBSET}")

    # ── Build loaders ─────────────────────────────────────────────────────────
    repeat_factors = compute_repeat_factors(train_records, cfg.NUM_CLASSES)
    sampler_train = RepeatFactorSampler(repeat_factors, seed=cfg.SEED)
    train_dataset = LVISMaskDataset(train_records, image_size=cfg.IMAGE_SIZE)
    val_dataset   = LVISMaskDataset(val_records, image_size=cfg.IMAGE_SIZE)
    eval_loader   = make_eval_loader(val_dataset)
    iters_per_epoch = max(1, len(train_records) // cfg.IMS_PER_BATCH)
    print(f"[Setup] iters_per_epoch: {iters_per_epoch}")


    # ── Pre-training sanity eval ──────────────────────────────────────────────
    # Evaluate the ImageNet-pretrained model BEFORE any LVIS training.
    # If this produces AP > 0, the eval pipeline is correct.
    print("\n[Sanity] Pre-training eval (ImageNet weights, no LVIS training) ...")
    pretrained_model = build_model(
        num_classes_with_bg=cfg.NUM_CLASSES + 1,
        freq_weight=cat_info["cat_freq_weight"],
        fed_loss_num_cat=cfg.FED_LOSS_NUM_CAT,
        use_fed_loss=cfg.USE_FED_LOSS,
        image_size=cfg.IMAGE_SIZE,
    ).to(DEVICE)
    metrics_pre = evaluate_on_lvis(pretrained_model, eval_loader, VAL_ANN_SUBSET, DEVICE)
    del pretrained_model; torch.cuda.empty_cache()

    print(f"[Sanity] Pre-training metrics: AP={metrics_pre.get('AP', 0):.4f}  "
          f"AP50={metrics_pre.get('AP50', 0):.4f}")
    if metrics_pre.get("AP", 0) > 0:
        print("[Sanity] ✅ Eval pipeline confirmed working (pre-training AP > 0)")
    else:
        print("[Sanity] ⚠ Pre-training AP = 0 (expected for untrained on 1203 classes)")
        print("[Sanity]   Training should fix this. If post-training AP is also 0,")
        print("[Sanity]   check that the val subset annotation JSON matches the eval subset.")


    # ── Baseline Training ─────────────────────────────────────────────────────
    final_ckpt = os.path.join(cfg.CKPT_DIR, "smoketest_final.pth")
    total_iters = cfg.BASELINE_EPOCHS * iters_per_epoch
    print(f"[Train] total_iters = {total_iters}")

    model = build_model(
        num_classes_with_bg=cfg.NUM_CLASSES + 1,
        freq_weight=cat_info["cat_freq_weight"],
        fed_loss_num_cat=cfg.FED_LOSS_NUM_CAT,
        use_fed_loss=cfg.USE_FED_LOSS,
        image_size=cfg.IMAGE_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda") if (cfg.USE_AMP and DEVICE.type == "cuda") else None

    loader = make_baseline_loader(train_dataset, sampler_train, iters_per_epoch)
    data_iter = iter(loader)
    logger = MetricLogger(window=cfg.LOG_EVERY)

    pbar = tqdm(range(total_iters), desc="[Baseline]")
    t_start = time.time()
    for step in pbar:
        try: batch = next(data_iter)
        except StopIteration: data_iter = iter(loader); batch = next(data_iter)
        lr = cosine_warmup_lr(step, cfg.LR, cfg.WARMUP_ITERS, total_iters)
        set_lr(optimizer, lr)
        losses = train_step(model, optimizer, scaler, batch, DEVICE, cfg.USE_AMP, cfg.CLIP_GRAD_NORM)
        logger.update(**losses, lr=lr)
        if step % cfg.LOG_EVERY == 0:
            pbar.set_postfix_str(logger.fmt(["loss_total", "lr"]))

    elapsed = time.time() - t_start
    torch.save({"model": model.state_dict(), "step": total_iters}, final_ckpt)
    print(f"[Train] DONE in {elapsed/60:.1f} min → {final_ckpt}")

    # ── Post-training Evaluation ──────────────────────────────────────────────
    del model; torch.cuda.empty_cache()

    eval_model = build_model(
        num_classes_with_bg=cfg.NUM_CLASSES + 1,
        freq_weight=cat_info["cat_freq_weight"],
        fed_loss_num_cat=cfg.FED_LOSS_NUM_CAT,
        use_fed_loss=cfg.USE_FED_LOSS,
        image_size=cfg.IMAGE_SIZE,
    ).to(DEVICE)
    state = torch.load(final_ckpt, map_location=DEVICE)
    eval_model.load_state_dict(state["model"], strict=False)

    print(f"[Eval] Running LVIS evaluation on {VAL_ANN_SUBSET} ...")
    metrics = evaluate_on_lvis(eval_model, eval_loader, VAL_ANN_SUBSET, DEVICE)
    del eval_model; torch.cuda.empty_cache()

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SMOKETEST RESULTS")
    print("=" * 60)
    print(f"  {'Metric':<8} {'Pre-train':>10} {'Post-train':>10} {'Δ':>8}")
    print("  " + "-" * 38)
    for k in ["AP", "AP50", "AP75", "APr", "APc", "APf"]:
        pre = metrics_pre.get(k, 0.0)
        post = metrics.get(k, 0.0)
        delta = post - pre
        print(f"  {k:<8} {pre:>10.4f} {post:>10.4f} {delta:>+8.4f}")
    print("=" * 60)

    ap = metrics.get("AP", 0.0)
    if ap > 0:
        print(f"\n✅ SMOKETEST PASSED: AP = {ap:.4f} (> 0)")
        print("   The pipeline produces non-zero metrics.")
        print("   Use bsgal.ipynb for full training with BSGAL + CB.")
    else:
        print(f"\n❌ SMOKETEST FAILED: AP = {ap:.4f}")
        print("   Possible causes:")
        print("   1. val subset annotation JSON doesn't match eval subset")
        print("   2. Dataset images missing from disk")
        print("   3. Model not producing any detections (check GPU)")

    # Print METRIC lines for automated parsing
    for k, v in metrics.items():
        print(f"METRIC {k}={v}")
    print(f"METRIC ap_improvement={metrics.get('AP', 0) - metrics_pre.get('AP', 0):.4f}")
    ''')
    )

    return nb


if __name__ == "__main__":
    nb = _build_notebook()
    out = "bsgal_smoketest.ipynb"
    with open(out, "w") as f:
        nbf.write(nb, f)
    print(f"Assembled {len(nb.cells)} cells → {out}")
