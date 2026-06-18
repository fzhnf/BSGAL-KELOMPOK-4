import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models import ResNet50_Weights

# CKPT_PATH = "checkpoints/bsgal/bsgal_final.pth"
CKPT_PATH = "models/bsgal-plain/bsgal_final.pth"
# CAT_INFO_JSON = "lvis_v1_train_cat_info.json"
CAT_INFO_JSON = "datasets/metadata/lvis_v1_train_cat_info.json"
IMAGE_PATH = "gambar_input.jpg"
IMAGE_PATH = "datasets/lvis/val2017/000000000139.jpg"
OUTPUT_PATH = "output.jpg"
SCORE_THRESH = 0.3
IMAGE_SIZE = 640
NUM_CLASSES = 1203
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cat_names(path: str, num_classes: int = 1203):
    with open(path) as f:
        items = json.load(f)
    items_sorted = sorted(items, key=lambda x: x["id"])
    return [items_sorted[i]["name"] for i in range(num_classes)]


cat_names = load_cat_names(CAT_INFO_JSON)


def build_model(num_classes_with_bg: int, image_size: int) -> nn.Module:
    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
        num_classes=num_classes_with_bg,
        min_size=image_size,
        max_size=image_size,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_test=500,
        box_detections_per_img=300,
        box_score_thresh=0.0,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes_with_bg
    )
    return model


model = build_model(NUM_CLASSES + 1, IMAGE_SIZE).to(DEVICE)
state = torch.load(CKPT_PATH, map_location=DEVICE)

# Fix key mismatch: checkpoint uses "box_predictor.predictor.cls_score" but
# FastRCNNPredictor expects "box_predictor.cls_score"
cpt = state["model"]
remapped = {}
for k, v in cpt.items():
    new_key = k.replace(
        "roi_heads.box_predictor.predictor.", "roi_heads.box_predictor."
    )
    remapped[new_key] = v

model.load_state_dict(remapped)
model.eval()
print(f"Model loaded from {CKPT_PATH}")


def letterbox(image: np.ndarray, size: int, pad_value: int = 114):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    image_r = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), pad_value, dtype=image.dtype)
    out[:new_h, :new_w] = image_r
    return out, scale, (size - new_w, size - new_h)


img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_lb, scale, (pad_x, pad_y) = letterbox(img_rgb, IMAGE_SIZE)
img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.to(DEVICE)


with torch.no_grad():
    preds = model([img_tensor])

pred = preds[0]
boxes = pred["boxes"].cpu().numpy()
labels = pred["labels"].cpu().numpy()
scores = pred["scores"].cpu().numpy()
masks = pred["masks"].cpu().numpy()

keep = scores >= SCORE_THRESH
boxes, labels, scores, masks = boxes[keep], labels[keep], scores[keep], masks[keep]
print(f"Terdeteksi {len(scores)} objek (threshold={SCORE_THRESH})")


orig_h, orig_w = img_rgb.shape[:2]


def to_orig(x1, y1, x2, y2, scale, pad_x, pad_y, orig_w, orig_h):
    x1 = np.clip(x1 / scale, 0, orig_w)
    y1 = np.clip(y1 / scale, 0, orig_h)
    x2 = np.clip(x2 / scale, 0, orig_w)
    y2 = np.clip(y2 / scale, 0, orig_h)
    return x1, y1, x2, y2


vis = img_rgb.copy()
rng = np.random.default_rng(42)
colors = rng.integers(80, 255, size=(NUM_CLASSES + 1, 3)).tolist()

for i in range(len(scores)):
    x1, y1, x2, y2 = boxes[i]
    x1, y1, x2, y2 = to_orig(x1, y1, x2, y2, scale, pad_x, pad_y, orig_w, orig_h)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    label_id = int(labels[i])
    name = cat_names[label_id - 1] if 1 <= label_id <= NUM_CLASSES else str(label_id)
    color = colors[label_id]

    m = (masks[i, 0] >= 0.5).astype(np.uint8)
    content_h = IMAGE_SIZE - pad_y
    content_w = IMAGE_SIZE - pad_x
    m_crop = m[:content_h, :content_w]
    m_orig = cv2.resize(m_crop, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    colored = np.zeros_like(vis)
    colored[m_orig == 1] = color
    vis = cv2.addWeighted(vis, 1.0, colored, 0.45, 0)

    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    label_text = f"{name} {scores[i]:.2f}"
    cv2.putText(
        vis,
        label_text,
        (x1, max(y1 - 5, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

cv2.imwrite(OUTPUT_PATH, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(f"Hasil disimpan ke {OUTPUT_PATH}")
