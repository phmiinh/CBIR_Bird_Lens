from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.measure import (
    label,
    moments_central,
    moments_hu,
    moments_normalized,
    regionprops,
)
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects,
)
from torch import nn
from torchvision import models

GLOBAL_HSV_BINS = (8, 8, 8)
REGIONAL_GRID = (2, 2)
LBP_POINTS = 8
LBP_RADIUS = 1
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

FEATURE_FILE_MAP = {
    "global_hsv_hist": "global_hsv_hist.npy",
    "regional_hsv_hist": "regional_hsv_hist.npy",
    "color_moments": "color_moments.npy",
    "lbp_hist": "lbp_hist.npy",
    "hog_descriptor": "hog_descriptor.npy",
    "silhouette_shape_descriptor": "silhouette_shape_descriptor.npy",
    "cnn_embedding": "cnn_embedding.npy",
}

FEATURE_SPECS = {
    "global_hsv_hist": {
        "display_name": "Global HSV Histogram",
        "similarity_metric": "chi_square_distance_then_inverse",
        "is_primary": 1,
        "information_value": "Overall color distribution of the bird image.",
        "strengths": "Robust for dominant plumage color and large color contrast.",
        "weaknesses": "Weak on spatial arrangement and fine texture.",
        "extraction_params": {"bins": list(GLOBAL_HSV_BINS)},
    },
    "regional_hsv_hist": {
        "display_name": "Regional HSV Histogram",
        "similarity_metric": "chi_square_distance_then_inverse",
        "is_primary": 1,
        "information_value": "Spatial color layout using a fixed 2x2 grid.",
        "strengths": "Captures color placement after normalization and side-view alignment.",
        "weaknesses": "Sensitive to crop shifts and local occlusion.",
        "extraction_params": {"bins": list(GLOBAL_HSV_BINS), "grid": list(REGIONAL_GRID)},
    },
    "color_moments": {
        "display_name": "Color Moments",
        "similarity_metric": "euclidean_distance_then_inverse",
        "is_primary": 1,
        "information_value": "Coarse HSV statistics via mean, standard deviation, and skewness.",
        "strengths": "Compact descriptor with good global color summary.",
        "weaknesses": "Too coarse for layout and local patterns.",
        "extraction_params": {"channels": ["H", "S", "V"], "moments": ["mean", "std", "skewness"]},
    },
    "lbp_hist": {
        "display_name": "LBP Histogram",
        "similarity_metric": "chi_square_distance_then_inverse",
        "is_primary": 1,
        "information_value": "Local micro-texture patterns in grayscale.",
        "strengths": "Useful for feather texture and repeated local patterns.",
        "weaknesses": "Weak for global shape and large-scale geometry.",
        "extraction_params": {"points": LBP_POINTS, "radius": LBP_RADIUS, "method": "uniform"},
    },
    "hog_descriptor": {
        "display_name": "HOG Descriptor",
        "similarity_metric": "cosine_similarity",
        "is_primary": 1,
        "information_value": "Edge structure, contour, and pose information.",
        "strengths": "Good for silhouette and body-part orientation.",
        "weaknesses": "Less robust to cluttered edges and weak contrast.",
        "extraction_params": {
            "orientations": HOG_ORIENTATIONS,
            "pixels_per_cell": list(HOG_PIXELS_PER_CELL),
            "cells_per_block": list(HOG_CELLS_PER_BLOCK),
            "block_norm": "L2-Hys",
        },
    },
    "silhouette_shape_descriptor": {
        "display_name": "Silhouette Shape Descriptor",
        "similarity_metric": "euclidean_distance_then_inverse",
        "is_primary": 1,
        "information_value": "Global bird silhouette geometry derived from the foreground mask.",
        "strengths": "Captures coarse body shape independently of dominant plumage color.",
        "weaknesses": "Depends on foreground-mask quality and does not encode fine texture.",
        "extraction_params": {
            "hu_moments": 7,
            "region_properties": [
                "area_ratio",
                "bbox_aspect_ratio_log1p",
                "extent",
                "eccentricity",
                "solidity",
                "orientation_normalized",
                "major_minor_ratio_log1p",
                "compactness",
            ],
        },
    },
    "cnn_embedding": {
        "display_name": "CNN Embedding",
        "similarity_metric": "cosine_similarity",
        "is_primary": 0,
        "information_value": "High-level semantic visual embedding from pretrained ResNet18.",
        "strengths": "Strong semantic abstraction and robust visual similarity.",
        "weaknesses": "Lower explainability; treated as secondary in this project.",
        "extraction_params": {"backbone": "resnet18", "pooling": "global_avg_pool"},
    },
}

DEFAULT_EXPERIMENTS = {
    "handcrafted_only": {
        "weights": {
            "regional_hsv_hist": 0.261,
            "global_hsv_hist": 0.094,
            "color_moments": 0.195,
            "lbp_hist": 0.118,
            "hog_descriptor": 0.272,
            "silhouette_shape_descriptor": 0.060,
        },
        "notes": "Primary explainable descriptor stack with explicit silhouette shape kept as a light auxiliary cue after tuning.",
    },
    "cnn_only": {
        "weights": {"cnn_embedding": 1.00},
        "notes": "Semantic baseline using only the secondary deep descriptor.",
    },
    "fusion": {
        "weights": {
            "regional_hsv_hist": 0.209,
            "global_hsv_hist": 0.075,
            "color_moments": 0.156,
            "lbp_hist": 0.094,
            "hog_descriptor": 0.218,
            "silhouette_shape_descriptor": 0.048,
            "cnn_embedding": 0.20,
        },
        "notes": "Handcrafted-first fusion after tuning the explicit silhouette shape cue down to a light auxiliary role.",
    },
    "ablation_no_regional_color": {
        "weights": {
            "global_hsv_hist": 0.127,
            "color_moments": 0.264,
            "lbp_hist": 0.159,
            "hog_descriptor": 0.367,
            "silhouette_shape_descriptor": 0.083,
        },
        "notes": "Ablation removing spatial color layout information.",
    },
    "ablation_no_explicit_shape": {
        "weights": {
            "regional_hsv_hist": 0.277,
            "global_hsv_hist": 0.100,
            "color_moments": 0.208,
            "lbp_hist": 0.125,
            "hog_descriptor": 0.290,
        },
        "notes": "Ablation removing the explicit silhouette shape descriptor while keeping color, texture, and contour.",
    },
    "ablation_no_shape": {
        "weights": {
            "regional_hsv_hist": 0.475,
            "global_hsv_hist": 0.170,
            "color_moments": 0.355,
        },
        "notes": "Ablation removing texture and shape descriptors.",
    },
}


def parse_bins(text: str) -> Tuple[int, int, int]:
    raw = str(text).strip()
    if not raw:
        return GLOBAL_HSV_BINS
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("HSV bins must have format h,s,v (example: 8,8,8).")
    values = tuple(int(part) for part in parts)
    if any(value <= 0 for value in values):
        raise ValueError("HSV bins must be positive integers.")
    return values


def load_csv_rows(file_path: Path) -> List[dict]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(file_path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(file_path: Path, payload: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_json(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_processed_image_path(row: dict, processed_root: Path) -> Path:
    relative_path = str(row.get("processed_relative_path", "")).strip()
    if not relative_path:
        raise ValueError("Missing processed_relative_path in metadata row.")

    direct_candidate = processed_root / relative_path
    if direct_candidate.exists():
        return direct_candidate

    images_candidate = processed_root / "images" / Path(relative_path).name
    if images_candidate.exists():
        return images_candidate

    raise FileNotFoundError(f"Cannot resolve processed image path: {relative_path}")


def _create_resnet(backbone: str) -> nn.Module:
    if backbone == "resnet18":
        try:
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            net = models.resnet18(pretrained=True)
    elif backbone == "resnet50":
        try:
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            net = models.resnet50(pretrained=True)
    else:
        raise ValueError("Unsupported backbone. Use resnet18 or resnet50.")

    feature_extractor = nn.Sequential(*list(net.children())[:-1])
    feature_extractor.eval()
    return feature_extractor


def select_device(device: str) -> torch.device:
    raw = str(device).strip().lower()
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if raw not in {"cpu", "cuda"}:
        raise ValueError("Device must be one of: auto, cpu, cuda.")
    return torch.device(raw)


def load_cnn_model(backbone: str, device: torch.device) -> nn.Module:
    model = _create_resnet(backbone)
    model.to(device)
    return model


def center_square_crop_resize(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    rgb = image.convert("RGB")
    side = min(rgb.width, rgb.height)
    left = max(0, (rgb.width - side) // 2)
    top = max(0, (rgb.height - side) // 2)
    crop = rgb.crop((left, top, left + side, top + side))
    return crop.resize(target_size, Image.Resampling.LANCZOS)


def compute_square_crop_from_bbox(
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    x, y, width, height = bbox
    center_x = x + (width / 2.0)
    center_y = y + (height / 2.0)
    side = max(width, height) * (1.0 + (2.0 * padding_ratio))
    side = max(1.0, min(side, float(min(image_width, image_height))))

    left = center_x - (side / 2.0)
    top = center_y - (side / 2.0)
    left = max(0.0, min(left, float(image_width) - side))
    top = max(0.0, min(top, float(image_height) - side))

    right = left + side
    bottom = top + side

    left_i = int(round(left))
    top_i = int(round(top))
    right_i = max(left_i + 1, min(int(round(right)), image_width))
    bottom_i = max(top_i + 1, min(int(round(bottom)), image_height))
    return left_i, top_i, right_i, bottom_i


def _estimate_foreground_bbox(image: Image.Image) -> Tuple[float, float, float, float] | None:
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    height, width = array.shape[:2]
    if min(height, width) < 32:
        return None

    border = max(8, int(round(min(height, width) * 0.08)))
    border_pixels = np.concatenate(
        [
            array[:border, :, :].reshape(-1, 3),
            array[-border:, :, :].reshape(-1, 3),
            array[:, :border, :].reshape(-1, 3),
            array[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_color = np.median(border_pixels, axis=0)
    color_distance = np.linalg.norm(array - background_color, axis=2)

    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
    saturation = hsv[:, :, 1]
    score_map = (0.8 * color_distance) + (0.2 * saturation)

    try:
        threshold = float(threshold_otsu(score_map))
    except ValueError:
        return None

    threshold = max(threshold, float(np.percentile(score_map, 70)))
    mask = score_map >= threshold
    radius = max(1, min(height, width) // 80)
    mask = binary_opening(mask, disk(radius))
    mask = binary_closing(mask, disk(radius))
    mask = remove_small_objects(mask, min_size=max(64, int(height * width * 0.01)))
    mask = remove_small_holes(mask, area_threshold=max(64, int(height * width * 0.01)))

    labeled = label(mask)
    if labeled.max() == 0:
        return None

    image_center_x = width / 2.0
    image_center_y = height / 2.0
    best_region = None
    best_score = -1.0
    for region in regionprops(labeled):
        min_row, min_col, max_row, max_col = region.bbox
        bbox_width = max_col - min_col
        bbox_height = max_row - min_row
        area = float(region.area)
        if bbox_width < width * 0.1 or bbox_height < height * 0.1:
            continue
        centroid_y, centroid_x = region.centroid
        normalized_center_distance = (
            ((centroid_x - image_center_x) / max(image_center_x, 1.0)) ** 2
            + ((centroid_y - image_center_y) / max(image_center_y, 1.0)) ** 2
        )
        region_score = area / (1.0 + normalized_center_distance)
        if region_score > best_score:
            best_score = region_score
            best_region = region

    if best_region is None:
        return None

    min_row, min_col, max_row, max_col = best_region.bbox
    bbox_width = max_col - min_col
    bbox_height = max_row - min_row
    return float(min_col), float(min_row), float(bbox_width), float(bbox_height)


def normalize_external_query_image(
    image: Image.Image,
    padding_ratio: float = 0.35,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[Image.Image, dict]:
    rgb = image.convert("RGB")
    if (rgb.width, rgb.height) == target_size:
        return rgb.copy(), {
            "method": "already_normalized_keep_as_is",
            "target_size": list(target_size),
        }

    estimated_bbox = _estimate_foreground_bbox(rgb)
    if estimated_bbox is None:
        normalized = center_square_crop_resize(rgb, target_size=target_size)
        return normalized, {
            "method": "center_square_crop_resize_fallback",
            "target_size": list(target_size),
        }

    crop_box = compute_square_crop_from_bbox(estimated_bbox, rgb.width, rgb.height, padding_ratio)
    normalized = rgb.crop(crop_box).resize(target_size, Image.Resampling.LANCZOS)
    return normalized, {
        "method": "estimated_bbox_square_crop_then_resize",
        "padding_ratio": float(padding_ratio),
        "target_size": list(target_size),
        "crop_box": list(crop_box),
        "estimated_bbox": [float(value) for value in estimated_bbox],
    }


def _sanitize_bbox_pixels(
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int] | None:
    image_width, image_height = image_size
    x, y, width, height = bbox
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(image_width, int(round(x + width)))
    bottom = min(image_height, int(round(y + height)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def project_bbox_to_normalized_mask(
    image_size: Tuple[int, int],
    bbox: Tuple[float, float, float, float],
    crop_box: Tuple[int, int, int, int],
) -> np.ndarray | None:
    image_width, image_height = image_size
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_width = max(1, int(crop_right - crop_left))
    crop_height = max(1, int(crop_bottom - crop_top))
    bbox_x, bbox_y, bbox_w, bbox_h = bbox

    normalized_bbox = (
        ((bbox_x - crop_left) * image_width) / crop_width,
        ((bbox_y - crop_top) * image_height) / crop_height,
        (bbox_w * image_width) / crop_width,
        (bbox_h * image_height) / crop_height,
    )
    pixel_bbox = _sanitize_bbox_pixels(normalized_bbox, image_size)
    if pixel_bbox is None:
        return None

    left, top, right, bottom = pixel_bbox
    mask = np.zeros((image_height, image_width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def estimate_foreground_mask(
    image: Image.Image,
    bbox_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    height, width = array.shape[:2]

    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
    border = max(6, int(round(min(height, width) * 0.06)))
    border_pixels = np.concatenate(
        [
            array[:border, :, :].reshape(-1, 3),
            array[-border:, :, :].reshape(-1, 3),
            array[:, :border, :].reshape(-1, 3),
            array[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_color = np.median(border_pixels, axis=0)
    color_distance = np.linalg.norm(array - background_color, axis=2)
    saturation = hsv[:, :, 1]
    score_map = (0.8 * color_distance) + (0.2 * saturation)

    try:
        threshold = float(threshold_otsu(score_map))
    except ValueError:
        threshold = float(np.percentile(score_map, 70))
    threshold = max(threshold, float(np.percentile(score_map, 70)))

    mask = score_map >= threshold
    radius = max(1, min(height, width) // 96)
    mask = binary_opening(mask, disk(radius))
    mask = binary_closing(mask, disk(radius))
    mask = remove_small_objects(mask, min_size=max(48, int(height * width * 0.005)))
    mask = remove_small_holes(mask, area_threshold=max(48, int(height * width * 0.005)))

    if bbox_mask is not None and np.any(bbox_mask):
        expanded_bbox_mask = binary_dilation(bbox_mask.astype(bool), disk(max(1, radius * 2)))
        constrained_mask = mask & expanded_bbox_mask
        if np.any(constrained_mask):
            mask = constrained_mask
        else:
            return bbox_mask.astype(bool), "bbox_rect_fallback"

    if np.any(mask):
        return mask.astype(bool), "estimated_foreground_mask"
    if bbox_mask is not None and np.any(bbox_mask):
        return bbox_mask.astype(bool), "bbox_rect_fallback"

    return np.ones((height, width), dtype=bool), "full_image_fallback"


def build_gallery_foreground_mask(image: Image.Image, row: dict) -> tuple[np.ndarray, str]:
    bbox = (
        float(row.get("bbox_x", 0.0)),
        float(row.get("bbox_y", 0.0)),
        float(row.get("bbox_w", 0.0)),
        float(row.get("bbox_h", 0.0)),
    )
    crop_box = (
        int(float(row.get("crop_left", 0))),
        int(float(row.get("crop_top", 0))),
        int(float(row.get("crop_right", image.width))),
        int(float(row.get("crop_bottom", image.height))),
    )
    bbox_mask = project_bbox_to_normalized_mask((image.width, image.height), bbox, crop_box)
    return estimate_foreground_mask(image, bbox_mask=bbox_mask)


def build_query_foreground_mask(image: Image.Image, preprocess_params: dict) -> tuple[np.ndarray, str]:
    bbox_payload = preprocess_params.get("estimated_bbox") or preprocess_params.get("bbox")
    crop_payload = preprocess_params.get("crop_box")
    bbox_mask = None
    if bbox_payload and crop_payload and len(bbox_payload) == 4 and len(crop_payload) == 4:
        bbox = tuple(float(value) for value in bbox_payload)
        crop_box = tuple(int(round(float(value))) for value in crop_payload)
        bbox_mask = project_bbox_to_normalized_mask((image.width, image.height), bbox, crop_box)
    return estimate_foreground_mask(image, bbox_mask=bbox_mask)


def preprocess_for_cnn(image: Image.Image, target_size: int = 224) -> torch.Tensor:
    rgb = center_square_crop_resize(image, target_size=(target_size, target_size))
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (array - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def extract_cnn_embedding(model: nn.Module, image: Image.Image, device: torch.device) -> np.ndarray:
    tensor = preprocess_for_cnn(image).to(device)
    with torch.no_grad():
        vector = model(tensor).flatten(start_dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    return vector


def _compute_hsv_histogram(
    hsv_image: np.ndarray,
    bins: Tuple[int, int, int],
    foreground_mask: np.ndarray | None = None,
) -> np.ndarray:
    if foreground_mask is not None and np.any(foreground_mask):
        pixels = hsv_image[foreground_mask]
        if pixels.size > 0:
            h_channel = pixels[:, 0]
            s_channel = pixels[:, 1]
            v_channel = pixels[:, 2]
        else:
            h_channel = hsv_image[:, :, 0].ravel()
            s_channel = hsv_image[:, :, 1].ravel()
            v_channel = hsv_image[:, :, 2].ravel()
    else:
        h_channel = hsv_image[:, :, 0].ravel()
        s_channel = hsv_image[:, :, 1].ravel()
        v_channel = hsv_image[:, :, 2].ravel()

    hist, _ = np.histogramdd(
        sample=(h_channel, s_channel, v_channel),
        bins=bins,
        range=((0, 256), (0, 256), (0, 256)),
    )
    vector = hist.astype(np.float32).ravel()
    denominator = float(vector.sum())
    if denominator > 0.0:
        vector /= denominator
    return vector


def extract_global_hsv_histogram(
    image: Image.Image,
    bins: Tuple[int, int, int],
    foreground_mask: np.ndarray | None = None,
) -> np.ndarray:
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    return _compute_hsv_histogram(hsv, bins=bins, foreground_mask=foreground_mask)


def extract_cumulative_hsv_histogram(
    image: Image.Image,
    bins: Tuple[int, int, int],
    foreground_mask: np.ndarray | None = None,
) -> np.ndarray:
    histogram = extract_global_hsv_histogram(image, bins=bins, foreground_mask=foreground_mask)
    cumulative = np.cumsum(histogram, dtype=np.float32)
    denominator = float(cumulative[-1]) if cumulative.size > 0 else 0.0
    if denominator > 0.0:
        cumulative /= denominator
    return cumulative.astype(np.float32)


def extract_regional_hsv_histogram(
    image: Image.Image,
    bins: Tuple[int, int, int],
    grid: Tuple[int, int] = REGIONAL_GRID,
    foreground_mask: np.ndarray | None = None,
) -> np.ndarray:
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    rows, cols = grid
    cell_vectors = []
    for row_index in range(rows):
        for col_index in range(cols):
            y0 = (row_index * hsv.shape[0]) // rows
            y1 = ((row_index + 1) * hsv.shape[0]) // rows
            x0 = (col_index * hsv.shape[1]) // cols
            x1 = ((col_index + 1) * hsv.shape[1]) // cols
            cell = hsv[y0:y1, x0:x1]
            cell_mask = None
            if foreground_mask is not None:
                cell_mask = foreground_mask[y0:y1, x0:x1]
            cell_vectors.append(_compute_hsv_histogram(cell, bins=bins, foreground_mask=cell_mask))

    vector = np.concatenate(cell_vectors).astype(np.float32)
    denominator = float(vector.sum())
    if denominator > 0.0:
        vector /= denominator
    return vector


def extract_color_moments(image: Image.Image, foreground_mask: np.ndarray | None = None) -> np.ndarray:
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    moments = []
    for channel_index in range(3):
        channel = hsv[:, :, channel_index]
        values = channel[foreground_mask] if foreground_mask is not None and np.any(foreground_mask) else channel.ravel()
        if values.size == 0:
            values = channel.ravel()
        mean = float(np.mean(values))
        std = float(np.std(values))
        centered = values - mean
        third_moment = float(np.mean(centered ** 3))
        skewness = np.sign(third_moment) * (abs(third_moment) ** (1.0 / 3.0))
        moments.extend([mean, std, skewness])
    return np.asarray(moments, dtype=np.float32)


def extract_lbp_histogram(
    image: Image.Image,
    points: int = LBP_POINTS,
    radius: int = LBP_RADIUS,
    foreground_mask: np.ndarray | None = None,
) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    lbp = local_binary_pattern(gray, P=points, R=radius, method="uniform")
    bins = np.arange(0, points + 3, dtype=np.float32)
    values = lbp[foreground_mask] if foreground_mask is not None and np.any(foreground_mask) else lbp.ravel()
    if values.size == 0:
        values = lbp.ravel()
    hist, _ = np.histogram(values, bins=bins, range=(0, points + 2))
    vector = hist.astype(np.float32)
    denominator = float(vector.sum())
    if denominator > 0.0:
        vector /= denominator
    return vector


def extract_hog_descriptor(
    image: Image.Image,
    foreground_mask: np.ndarray | None = None,
    mask_method: str = "full_image_fallback",
) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if foreground_mask is not None and np.any(foreground_mask) and mask_method == "estimated_foreground_mask":
        gray = gray * foreground_mask.astype(np.float32)
    vector = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return np.asarray(vector, dtype=np.float32)


def _prepare_shape_mask(foreground_mask: np.ndarray | None, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if foreground_mask is None or not np.any(foreground_mask):
        return np.ones((height, width), dtype=bool)

    mask = foreground_mask.astype(bool)
    mask = binary_closing(mask, disk(1))
    mask = remove_small_holes(mask, area_threshold=max(16, int(mask.size * 0.001)))

    labeled = label(mask)
    if labeled.max() == 0:
        return np.ones((height, width), dtype=bool)

    largest_region = max(regionprops(labeled), key=lambda region: float(region.area))
    component_mask = labeled == largest_region.label
    return component_mask.astype(bool)


def extract_silhouette_shape_descriptor(
    image: Image.Image,
    foreground_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = _prepare_shape_mask(foreground_mask, image.size)
    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        raise ValueError("Unable to derive a valid foreground region for silhouette shape extraction.")

    region = max(regions, key=lambda candidate: float(candidate.area))
    component_mask = (labeled == region.label).astype(np.float32)

    hu = moments_hu(moments_normalized(moments_central(component_mask)))
    hu = np.sign(hu) * np.log1p(np.abs(hu))

    min_row, min_col, max_row, max_col = region.bbox
    bbox_height = max(1.0, float(max_row - min_row))
    bbox_width = max(1.0, float(max_col - min_col))
    area = float(region.area)
    image_area = float(mask.shape[0] * mask.shape[1])
    aspect_ratio = bbox_width / bbox_height
    major_minor_ratio = float(region.major_axis_length) / max(float(region.minor_axis_length), 1e-6)
    perimeter = max(float(region.perimeter), 1e-6)
    compactness = float((4.0 * np.pi * area) / (perimeter * perimeter))
    orientation = float(region.orientation) / (np.pi / 2.0)

    geometric_features = np.asarray(
        [
            area / max(image_area, 1.0),
            np.log1p(aspect_ratio),
            float(region.extent),
            float(region.eccentricity),
            float(region.solidity),
            float(np.clip(orientation, -1.0, 1.0)),
            np.log1p(major_minor_ratio),
            compactness,
        ],
        dtype=np.float32,
    )
    return np.concatenate([hu.astype(np.float32), geometric_features], axis=0)


def extract_all_features(
    image: Image.Image,
    bins: Tuple[int, int, int],
    regional_grid: Tuple[int, int],
    cnn_model: nn.Module | None,
    device: torch.device | None,
    foreground_mask: np.ndarray | None = None,
    mask_method: str = "full_image_fallback",
) -> Dict[str, np.ndarray]:
    rgb = image.convert("RGB")
    features = {
        "global_hsv_hist": extract_global_hsv_histogram(rgb, bins=bins, foreground_mask=foreground_mask),
        "regional_hsv_hist": extract_regional_hsv_histogram(
            rgb,
            bins=bins,
            grid=regional_grid,
            foreground_mask=foreground_mask,
        ),
        "color_moments": extract_color_moments(rgb, foreground_mask=foreground_mask),
        "lbp_hist": extract_lbp_histogram(rgb, foreground_mask=foreground_mask),
        "hog_descriptor": extract_hog_descriptor(rgb, foreground_mask=foreground_mask, mask_method=mask_method),
        "silhouette_shape_descriptor": extract_silhouette_shape_descriptor(
            rgb,
            foreground_mask=foreground_mask,
        ),
    }
    if cnn_model is not None and device is not None:
        features["cnn_embedding"] = extract_cnn_embedding(cnn_model, rgb, device)
    return features


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def cosine_similarity_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / max(float(np.linalg.norm(query)), 1e-12)
    matrix_norm = l2_normalize(matrix)
    return matrix_norm @ query_norm


def chi_square_distance(query: np.ndarray, matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    numerator = (matrix - query) ** 2
    denominator = matrix + query + eps
    return 0.5 * np.sum(numerator / denominator, axis=1)


def euclidean_distance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.linalg.norm(matrix - query, axis=1)


def inverse_distance_similarity(distance: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + distance)


def chi_square_similarity_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return inverse_distance_similarity(chi_square_distance(query, matrix))


def euclidean_similarity_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return inverse_distance_similarity(euclidean_distance(query, matrix))


def index_by_image_id(rows: List[dict]) -> Dict[int, int]:
    index: Dict[int, int] = {}
    for item_index, row in enumerate(rows):
        index[int(row["image_id"])] = item_index
    return index


def load_feature_arrays(feature_dir: Path, feature_names: Iterable[str]) -> Dict[str, np.ndarray]:
    arrays = {}
    for name in feature_names:
        file_name = FEATURE_FILE_MAP[name]
        arrays[name] = np.load(feature_dir / file_name)
    return arrays


def build_descriptor_table_rows(feature_dims: Dict[str, int]) -> List[dict]:
    rows = []
    for name, spec in FEATURE_SPECS.items():
        rows.append(
            {
                "feature_name": name,
                "display_name": spec["display_name"],
                "vector_dim": feature_dims.get(name, 0),
                "similarity_metric": spec["similarity_metric"],
                "is_primary": spec["is_primary"],
                "information_value": spec["information_value"],
                "strengths": spec["strengths"],
                "weaknesses": spec["weaknesses"],
            }
        )
    return rows


def build_feature_type_rows(feature_dims: Dict[str, int]) -> List[dict]:
    rows = []
    for feature_type_id, name in enumerate(FEATURE_SPECS.keys(), start=1):
        spec = FEATURE_SPECS[name]
        rows.append(
            {
                "feature_type_id": feature_type_id,
                "name": name,
                "vector_dim": int(feature_dims[name]),
                "similarity_metric": spec["similarity_metric"],
                "extraction_params_json": json.dumps(spec["extraction_params"], ensure_ascii=True),
                "is_primary": int(spec["is_primary"]),
            }
        )
    return rows


def build_experiment_rows(dataset_version: str) -> List[dict]:
    rows = []
    for experiment_index, (name, config) in enumerate(DEFAULT_EXPERIMENTS.items(), start=1):
        rows.append(
            {
                "experiment_id": experiment_index,
                "name": name,
                "feature_set_json": json.dumps(sorted(config["weights"].keys()), ensure_ascii=True),
                "weighting_json": json.dumps(config["weights"], ensure_ascii=True),
                "dataset_version": dataset_version,
                "summary_metrics_json": json.dumps({}, ensure_ascii=True),
                "notes": config["notes"],
            }
        )
    return rows
