from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw

REQUIRED_FILES = (
    "images.txt",
    "bounding_boxes.txt",
    "image_class_labels.txt",
    "classes.txt",
    "train_test_split.txt",
)


def locate_cub_root(base_dir: Path) -> Path:
    base_dir = base_dir.resolve()
    candidates = []

    for images_file in base_dir.rglob("images.txt"):
        parent = images_file.parent
        if all((parent / file_name).exists() for file_name in REQUIRED_FILES):
            candidates.append(parent)

    if all((base_dir / file_name).exists() for file_name in REQUIRED_FILES):
        candidates.append(base_dir)

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate a CUB-200-2011 root inside {base_dir}. "
            "Expected images.txt, bounding_boxes.txt, image_class_labels.txt, "
            "classes.txt, and train_test_split.txt."
        )

    candidates.sort(key=lambda path: (len(path.parts), len(str(path))))
    return candidates[0]


def _read_simple_mapping(file_path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            key_text, value = line.split(" ", 1)
            mapping[int(key_text)] = value
    return mapping


def _read_bbox_mapping(file_path: Path) -> Dict[int, Tuple[float, float, float, float]]:
    mapping: Dict[int, Tuple[float, float, float, float]] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            image_id, x, y, width, height = line.split()
            mapping[int(image_id)] = (float(x), float(y), float(width), float(height))
    return mapping


def _resolve_image_path(dataset_root: Path, relative_image_path: str) -> Path:
    images_dir_candidate = dataset_root / "images" / relative_image_path
    if images_dir_candidate.exists():
        return images_dir_candidate

    direct_candidate = dataset_root / relative_image_path
    if direct_candidate.exists():
        return direct_candidate

    raise FileNotFoundError(
        f"Image {relative_image_path} was referenced by metadata but not found under {dataset_root}"
    )


def load_cub_records(dataset_root: Path) -> List[dict]:
    dataset_root = locate_cub_root(dataset_root)

    image_paths = _read_simple_mapping(dataset_root / "images.txt")
    class_labels_raw = _read_simple_mapping(dataset_root / "image_class_labels.txt")
    class_names_raw = _read_simple_mapping(dataset_root / "classes.txt")
    split_raw = _read_simple_mapping(dataset_root / "train_test_split.txt")
    bboxes = _read_bbox_mapping(dataset_root / "bounding_boxes.txt")

    records: List[dict] = []
    for image_id in sorted(image_paths):
        class_id = int(class_labels_raw[image_id])
        split_value = int(split_raw[image_id])
        relative_image_path = image_paths[image_id]
        image_path = _resolve_image_path(dataset_root, relative_image_path)
        bbox_x, bbox_y, bbox_w, bbox_h = bboxes[image_id]

        records.append(
            {
                "image_id": image_id,
                "relative_image_path": relative_image_path,
                "absolute_image_path": str(image_path),
                "species_id": class_id,
                "species_name": class_names_raw[class_id],
                "split": "train" if split_value == 1 else "test",
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
            }
        )

    return records


def load_attribute_names(dataset_root: Path) -> Dict[int, str]:
    dataset_root = locate_cub_root(dataset_root)
    return _read_simple_mapping(dataset_root / "attributes" / "attributes.txt")


def load_image_attribute_presence(
    dataset_root: Path,
    attribute_ids: Iterable[int],
    use_clean_file: bool = True,
) -> Dict[int, Dict[int, int]]:
    dataset_root = locate_cub_root(dataset_root)
    selected_ids = {int(attribute_id) for attribute_id in attribute_ids}
    file_name = "image_attribute_labels_clean.txt" if use_clean_file else "image_attribute_labels.txt"
    file_path = dataset_root / "attributes" / file_name

    result: Dict[int, Dict[int, int]] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            image_id_text, attribute_id_text, is_present_text, *_ = line.split()
            attribute_id = int(attribute_id_text)
            if attribute_id not in selected_ids:
                continue
            image_id = int(image_id_text)
            result.setdefault(image_id, {})[attribute_id] = int(is_present_text)
    return result


def load_part_visibility(
    dataset_root: Path,
    part_ids: Iterable[int],
) -> Dict[int, Dict[int, int]]:
    dataset_root = locate_cub_root(dataset_root)
    selected_ids = {int(part_id) for part_id in part_ids}
    file_path = dataset_root / "parts" / "part_locs.txt"

    result: Dict[int, Dict[int, int]] = {}
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            image_id_text, part_id_text, _x, _y, visible_text = line.split()
            part_id = int(part_id_text)
            if part_id not in selected_ids:
                continue
            image_id = int(image_id_text)
            result.setdefault(image_id, {})[part_id] = int(visible_text)
    return result


def compute_square_crop(
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

    crop_box = (
        int(round(left)),
        int(round(top)),
        int(round(right)),
        int(round(bottom)),
    )

    left_i, top_i, right_i, bottom_i = crop_box
    right_i = max(left_i + 1, min(right_i, image_width))
    bottom_i = max(top_i + 1, min(bottom_i, image_height))
    return left_i, top_i, right_i, bottom_i


def normalize_image(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    padding_ratio: float,
    target_size: Tuple[int, int],
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    crop_box = compute_square_crop(bbox, image.width, image.height, padding_ratio)
    cropped = image.crop(crop_box)
    resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
    return resized, crop_box


def draw_bbox(image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    x, y, width, height = bbox
    stroke = max(2, min(image.size) // 100)
    draw.rectangle((x, y, x + width, y + height), outline="red", width=stroke)
    return annotated


def write_csv(file_path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
