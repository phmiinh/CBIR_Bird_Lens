from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models


def parse_bins(text: str) -> Tuple[int, int, int]:
    raw = str(text).strip()
    if not raw:
        return 8, 8, 8
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


def preprocess_for_cnn(image: Image.Image, target_size: int = 224) -> torch.Tensor:
    rgb = image.convert("RGB").resize((target_size, target_size), Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (array - mean) / std
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor


def extract_cnn_embedding(model: nn.Module, image: Image.Image, device: torch.device) -> np.ndarray:
    tensor = preprocess_for_cnn(image).to(device)
    with torch.no_grad():
        vector = model(tensor).flatten(start_dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    return vector


def extract_hsv_histogram(image: Image.Image, bins: Tuple[int, int, int]) -> np.ndarray:
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    hist, _ = np.histogramdd(
        sample=(hsv[:, :, 0].ravel(), hsv[:, :, 1].ravel(), hsv[:, :, 2].ravel()),
        bins=bins,
        range=((0, 256), (0, 256), (0, 256)),
    )
    vector = hist.astype(np.float32).ravel()
    denominator = float(vector.sum())
    if denominator > 0.0:
        vector /= denominator
    return vector


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def cosine_similarity_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / max(float(np.linalg.norm(query)), 1e-12)
    matrix_norm = l2_normalize(matrix)
    return matrix_norm @ query_norm


def chi_square_similarity_scores(query: np.ndarray, matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    numerator = (matrix - query) ** 2
    denominator = matrix + query + eps
    distance = 0.5 * np.sum(numerator / denominator, axis=1)
    return 1.0 / (1.0 + distance)


def save_json(file_path: Path, payload: dict) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_json(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def index_by_image_id(rows: List[dict]) -> Dict[int, int]:
    index: Dict[int, int] = {}
    for i, row in enumerate(rows):
        image_id = int(row["image_id"])
        index[image_id] = i
    return index
