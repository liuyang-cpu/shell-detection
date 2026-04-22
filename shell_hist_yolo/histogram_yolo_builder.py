from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VALID_STATS = ("mean", "iqr", "p10", "p90", "foreground_ratio")


@dataclass(slots=True)
class HistogramYOLOConfig:
    background_threshold: int = 250
    foreground_mode: str = "smart"
    hist_bins: int = 16
    hist_mode: str = "object"
    merge_mode: str = "mean"
    include_stats: tuple[str, ...] = ("mean", "iqr", "p10", "p90")
    save_format: str = "npy"
    skip_empty_labels: bool = False

    def validate(self) -> None:
        if not 1 <= self.background_threshold <= 255:
            raise ValueError("background_threshold must be in [1, 255]")
        if self.foreground_mode not in {"smart", "threshold"}:
            raise ValueError("foreground_mode must be 'smart' or 'threshold'")
        if self.hist_bins <= 0:
            raise ValueError("hist_bins must be > 0")
        if self.hist_mode not in {"object", "global"}:
            raise ValueError("hist_mode must be 'object' or 'global'")
        if self.merge_mode not in {"mean", "sum", "max"}:
            raise ValueError("merge_mode must be one of: mean, sum, max")
        if self.save_format not in {"npy", "pt", "both"}:
            raise ValueError("save_format must be one of: npy, pt, both")
        unknown_stats = sorted(set(self.include_stats) - set(VALID_STATS))
        if unknown_stats:
            raise ValueError(f"unsupported stats: {unknown_stats}")


@dataclass(slots=True)
class BBoxRecord:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(slots=True)
class ObjectFeatures:
    bbox: BBoxRecord
    histogram: np.ndarray
    stats: dict[str, float]
    foreground_count: int


@dataclass(slots=True)
class BuildResult:
    array: np.ndarray
    image_path: Path
    label_path: Path
    object_count: int
    metadata: dict[str, object] = field(default_factory=dict)


class HistogramYOLOBuilder:
    def __init__(self, config: HistogramYOLOConfig | None = None) -> None:
        self.config = config or HistogramYOLOConfig()
        self.config.validate()

    @property
    def channel_count(self) -> int:
        return 1 + self.config.hist_bins + len(self.config.include_stats)

    def process_image(self, image_path: Path, label_path: Path) -> BuildResult:
        image = self._load_grayscale_image(image_path)
        height, width = image.shape
        boxes = self._load_yolo_boxes(label_path, width=width, height=height)

        if self.config.skip_empty_labels and not boxes:
            raise ValueError(f"no valid labels found in {label_path}")

        grayscale_channel = image.astype(np.float32) / 255.0
        object_features = [self._extract_object_features(image, bbox) for bbox in boxes]

        histogram_channels = self._build_histogram_channels(
            image=image,
            object_features=object_features,
        )
        stat_channels = self._build_stat_channels(
            image_shape=image.shape,
            object_features=object_features,
        )

        channels = [grayscale_channel]
        channels.extend(histogram_channels)
        channels.extend(stat_channels)
        # Ultralytics augmentation pipeline expects image-like arrays in HWC layout.
        stacked = np.stack(channels, axis=-1).astype(np.float32)

        metadata = {
            "image_path": str(image_path),
            "label_path": str(label_path),
            "shape": list(stacked.shape),
            "height": height,
            "width": width,
            "object_count": len(object_features),
            "hist_bins": self.config.hist_bins,
            "hist_mode": self.config.hist_mode,
            "merge_mode": self.config.merge_mode,
            "background_threshold": self.config.background_threshold,
            "foreground_mode": self.config.foreground_mode,
            "include_stats": list(self.config.include_stats),
            "objects": [
                {
                    "class_id": item.bbox.class_id,
                    "bbox_xyxy": [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2],
                    "foreground_count": item.foreground_count,
                    "stats": item.stats,
                }
                for item in object_features
            ],
        }
        return BuildResult(
            array=stacked,
            image_path=image_path,
            label_path=label_path,
            object_count=len(object_features),
            metadata=metadata,
        )

    def process_dataset_root(
        self,
        datasets_root: Path,
        output_root: Path,
        splits: Iterable[str] = ("train", "val"),
        dataset_names: Iterable[str] | None = None,
    ) -> dict[str, object]:
        datasets_root = Path(datasets_root)
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        dataset_dirs = self._discover_dataset_dirs(datasets_root, dataset_names=dataset_names)
        summary: dict[str, object] = {
            "datasets_root": str(datasets_root),
            "output_root": str(output_root),
            "config": asdict(self.config),
            "datasets": {},
        }

        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name if dataset_dir != datasets_root else "root_dataset"
            dataset_output_dir = output_root / dataset_name
            dataset_summary: dict[str, object] = {
                "dataset_dir": str(dataset_dir),
                "output_dir": str(dataset_output_dir),
                "channels": self.channel_count,
                "tensor_format": self.config.save_format,
                "splits": {},
                "processed_images": 0,
                "class_ids": [],
            }
            dataset_class_ids: set[int] = set()

            for split in splits:
                split_summary = self._process_split(
                    dataset_dir=dataset_dir,
                    dataset_output_dir=dataset_output_dir,
                    split=split,
                )
                dataset_summary["splits"][split] = split_summary
                dataset_summary["processed_images"] += int(split_summary["processed_images"])
                dataset_class_ids.update(int(class_id) for class_id in split_summary["class_ids"])

            dataset_summary["class_ids"] = sorted(dataset_class_ids)
            self._write_dataset_files(
                dataset_name=dataset_name,
                dataset_output_dir=dataset_output_dir,
                split_names=[name for name, value in dataset_summary["splits"].items() if value["processed_images"] > 0],
                class_ids=sorted(dataset_class_ids),
            )

            summary["datasets"][dataset_name] = dataset_summary

        metadata_path = output_root / "metadata.json"
        metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def _process_split(self, dataset_dir: Path, dataset_output_dir: Path, split: str) -> dict[str, object]:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            return {
                "processed_images": 0,
                "image_dir": str(image_dir),
                "label_dir": str(label_dir),
                "tensor_dir": str(dataset_output_dir / "images" / split),
                "sidecar_dir": str(dataset_output_dir / "sidecar" / split),
                "output_label_dir": str(dataset_output_dir / "labels" / split),
                "class_ids": [],
            }

        image_paths = sorted(
            (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS),
            key=lambda path: path.name,
        )

        processed = 0
        split_class_ids: set[int] = set()
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue

            try:
                result = self.process_image(image_path=image_path, label_path=label_path)
            except ValueError:
                continue

            self._save_result(result=result, dataset_output_dir=dataset_output_dir, split=split)
            processed += 1
            split_class_ids.update(int(item["class_id"]) for item in result.metadata["objects"])

        return {
            "processed_images": processed,
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "tensor_dir": str(dataset_output_dir / "images" / split),
            "sidecar_dir": str(dataset_output_dir / "sidecar" / split),
            "output_label_dir": str(dataset_output_dir / "labels" / split),
            "class_ids": sorted(split_class_ids),
        }

    def _save_result(self, result: BuildResult, dataset_output_dir: Path, split: str) -> None:
        image_output_dir = dataset_output_dir / "images" / split
        label_output_dir = dataset_output_dir / "labels" / split
        sidecar_output_dir = dataset_output_dir / "sidecar" / split
        image_output_dir.mkdir(parents=True, exist_ok=True)
        label_output_dir.mkdir(parents=True, exist_ok=True)
        sidecar_output_dir.mkdir(parents=True, exist_ok=True)
        stem = result.image_path.stem

        if self.config.save_format in {"npy", "both"}:
            np.save(image_output_dir / f"{stem}.npy", result.array)

        if self.config.save_format in {"pt", "both"}:
            if torch is None:
                raise RuntimeError("save_format requires torch, but torch is not installed")
            torch.save(torch.from_numpy(result.array), image_output_dir / f"{stem}.pt")

        shutil.copy2(result.label_path, label_output_dir / f"{stem}.txt")

        sidecar_path = sidecar_output_dir / f"{stem}.json"
        sidecar_path.write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_dataset_files(
        self,
        dataset_name: str,
        dataset_output_dir: Path,
        split_names: list[str],
        class_ids: list[int],
    ) -> None:
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        dataset_metadata = {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_output_dir.resolve()),
            "channels": self.channel_count,
            "save_format": self.config.save_format,
            "hist_bins": self.config.hist_bins,
            "hist_mode": self.config.hist_mode,
            "merge_mode": self.config.merge_mode,
            "foreground_mode": self.config.foreground_mode,
            "background_threshold": self.config.background_threshold,
            "include_stats": list(self.config.include_stats),
            "splits": split_names,
            "class_ids": class_ids,
            "names": {class_id: f"class_{class_id}" for class_id in class_ids},
        }
        metadata_path = dataset_output_dir / "dataset_metadata.json"
        metadata_path.write_text(json.dumps(dataset_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        if not split_names:
            return

        yaml_lines = [f"path: {dataset_output_dir.resolve().as_posix()}"]
        for split_name in ("train", "val", "test"):
            if split_name in split_names:
                yaml_lines.append(f"{split_name}: images/{split_name}")
        yaml_lines.append("")
        yaml_lines.append(f"channels: {self.channel_count}")
        yaml_lines.append("")
        yaml_lines.append("names:")
        if class_ids:
            for class_id in class_ids:
                yaml_lines.append(f"  {class_id}: class_{class_id}")
        else:
            yaml_lines.append("  0: class_0")
        yaml_lines.append("")
        yaml_lines.append(f"# tensor_format: {self.config.save_format}")
        yaml_lines.append("# Update class names above before training if you have semantic labels.")
        (dataset_output_dir / "dataset.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    def _load_grayscale_image(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)

    def _load_yolo_boxes(self, label_path: Path, width: int, height: int) -> list[BBoxRecord]:
        boxes: list[BBoxRecord] = []
        lines = label_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, box_w, box_h = parts
            try:
                bbox = self._yolo_to_xyxy(
                    class_id=int(float(class_id)),
                    x_center=float(x_center),
                    y_center=float(y_center),
                    width_norm=float(box_w),
                    height_norm=float(box_h),
                    image_width=width,
                    image_height=height,
                )
            except ValueError:
                continue

            boxes.append(bbox)
        return boxes

    def _yolo_to_xyxy(
        self,
        class_id: int,
        x_center: float,
        y_center: float,
        width_norm: float,
        height_norm: float,
        image_width: int,
        image_height: int,
    ) -> BBoxRecord:
        bw = width_norm * image_width
        bh = height_norm * image_height
        cx = x_center * image_width
        cy = y_center * image_height

        x1 = max(0, int(np.floor(cx - bw / 2)))
        y1 = max(0, int(np.floor(cy - bh / 2)))
        x2 = min(image_width, int(np.ceil(cx + bw / 2)))
        y2 = min(image_height, int(np.ceil(cy + bh / 2)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("invalid bbox")

        return BBoxRecord(
            class_id=class_id,
            x_center=x_center,
            y_center=y_center,
            width=width_norm,
            height=height_norm,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

    def _extract_object_features(self, image: np.ndarray, bbox: BBoxRecord) -> ObjectFeatures:
        crop = image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        foreground_mask = self._extract_foreground_mask(crop)
        foreground = crop[foreground_mask]

        if foreground.size == 0:
            histogram = np.zeros(self.config.hist_bins, dtype=np.float32)
            stats = {name: 0.0 for name in VALID_STATS}
            return ObjectFeatures(bbox=bbox, histogram=histogram, stats=stats, foreground_count=0)

        histogram = self._normalized_histogram(foreground)
        stats = self._compute_stats(foreground, crop.size)
        return ObjectFeatures(
            bbox=bbox,
            histogram=histogram,
            stats=stats,
            foreground_count=int(foreground.size),
        )

    def _extract_foreground_mask(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros_like(crop, dtype=bool)

        if self.config.foreground_mode == "threshold":
            return crop < self.config.background_threshold

        mask = self._extract_smart_foreground_mask(crop)
        if np.any(mask):
            return mask
        return crop < self.config.background_threshold

    def _extract_smart_foreground_mask(self, crop: np.ndarray) -> np.ndarray:
        threshold = self._otsu_threshold(crop)
        candidate = crop <= threshold

        if not np.any(candidate):
            return np.zeros_like(candidate, dtype=bool)

        components = self._connected_components(candidate)
        if not components:
            return np.zeros_like(candidate, dtype=bool)

        height, width = crop.shape
        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0
        norm = max(float(np.hypot(center_y, center_x)), 1.0)

        best_mask = None
        best_score = -1.0
        min_area = max(16, int(crop.size * 0.01))
        for component in components:
            area = int(component.sum())
            if area < min_area:
                continue

            ys, xs = np.nonzero(component)
            comp_center_y = float(ys.mean())
            comp_center_x = float(xs.mean())
            center_distance = np.hypot(comp_center_y - center_y, comp_center_x - center_x) / norm
            center_weight = max(0.25, 1.0 - 0.5 * center_distance)
            score = area * center_weight
            if score > best_score:
                best_score = score
                best_mask = component

        if best_mask is not None:
            return best_mask

        return max(components, key=lambda item: int(item.sum()))

    def _otsu_threshold(self, values: np.ndarray) -> int:
        flat = values.astype(np.uint8).reshape(-1)
        if flat.size == 0:
            return self.config.background_threshold - 1

        histogram = np.bincount(flat, minlength=256).astype(np.float64)
        total = histogram.sum()
        if total <= 0:
            return self.config.background_threshold - 1

        probability = histogram / total
        omega = np.cumsum(probability)
        mu = np.cumsum(probability * np.arange(256, dtype=np.float64))
        mu_t = mu[-1]

        denominator = omega * (1.0 - omega)
        sigma_b2 = np.zeros(256, dtype=np.float64)
        valid = denominator > 0
        sigma_b2[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denominator[valid]
        return int(np.argmax(sigma_b2))

    def _connected_components(self, mask: np.ndarray) -> list[np.ndarray]:
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        components: list[np.ndarray] = []

        for y in range(height):
            for x in range(width):
                if not mask[y, x] or visited[y, x]:
                    continue

                stack = [(y, x)]
                visited[y, x] = True
                coords: list[tuple[int, int]] = []

                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                component = np.zeros_like(mask, dtype=bool)
                ys, xs = zip(*coords)
                component[np.asarray(ys), np.asarray(xs)] = True
                components.append(component)

        return components

    def _normalized_histogram(self, foreground: np.ndarray) -> np.ndarray:
        histogram, _ = np.histogram(foreground, bins=self.config.hist_bins, range=(0, 256))
        histogram = histogram.astype(np.float32)
        total = float(histogram.sum())
        if total > 0:
            histogram /= total
        return histogram

    def _compute_stats(self, foreground: np.ndarray, crop_area: int) -> dict[str, float]:
        q1 = float(np.percentile(foreground, 25))
        q3 = float(np.percentile(foreground, 75))
        stats = {
            "mean": float(np.mean(foreground)) / 255.0,
            "iqr": max(0.0, (q3 - q1) / 255.0),
            "p10": float(np.percentile(foreground, 10)) / 255.0,
            "p90": float(np.percentile(foreground, 90)) / 255.0,
            "foreground_ratio": float(foreground.size) / float(max(crop_area, 1)),
        }
        return stats

    def _build_histogram_channels(
        self,
        image: np.ndarray,
        object_features: list[ObjectFeatures],
    ) -> list[np.ndarray]:
        height, width = image.shape
        if self.config.hist_mode == "global":
            return self._build_global_histogram_channels(image=image, object_features=object_features)

        accum = np.zeros((self.config.hist_bins, height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)

        for item in object_features:
            x1, y1, x2, y2 = item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2
            patch = self._histogram_to_patch(item.histogram, height=y2 - y1, width=x2 - x1)
            if self.config.merge_mode == "max":
                accum[:, y1:y2, x1:x2] = np.maximum(accum[:, y1:y2, x1:x2], patch)
            else:
                accum[:, y1:y2, x1:x2] += patch
                counts[y1:y2, x1:x2] += 1.0

        return self._finalize_feature_stack(accum=accum, counts=counts, already_merged=self.config.merge_mode == "max")

    def _build_global_histogram_channels(
        self,
        image: np.ndarray,
        object_features: list[ObjectFeatures],
    ) -> list[np.ndarray]:
        height, width = image.shape
        accum = np.zeros((self.config.hist_bins, height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)

        all_foregrounds = []
        for item in object_features:
            if item.foreground_count > 0:
                all_foregrounds.append(item.histogram * item.foreground_count)

        if all_foregrounds:
            merged = np.sum(np.stack(all_foregrounds, axis=0), axis=0)
            total = merged.sum()
            if total > 0:
                merged /= total
        else:
            merged = np.zeros(self.config.hist_bins, dtype=np.float32)

        patch = self._histogram_to_patch(merged, height=height, width=width)
        accum += patch
        counts += 1.0
        return self._finalize_feature_stack(accum=accum, counts=counts, already_merged=False)

    def _build_stat_channels(
        self,
        image_shape: tuple[int, int],
        object_features: list[ObjectFeatures],
    ) -> list[np.ndarray]:
        height, width = image_shape
        channels: list[np.ndarray] = []

        for stat_name in self.config.include_stats:
            accum = np.zeros((height, width), dtype=np.float32)
            counts = np.zeros((height, width), dtype=np.float32)

            for item in object_features:
                x1, y1, x2, y2 = item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2
                accum[y1:y2, x1:x2] += item.stats[stat_name]
                counts[y1:y2, x1:x2] += 1.0

            if self.config.merge_mode == "mean":
                valid = counts > 0
                channel = np.zeros((height, width), dtype=np.float32)
                channel[valid] = accum[valid] / counts[valid]
            elif self.config.merge_mode == "sum":
                channel = accum
            else:
                channel = np.zeros((height, width), dtype=np.float32)
                for item in object_features:
                    x1, y1, x2, y2 = item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2
                    channel[y1:y2, x1:x2] = np.maximum(channel[y1:y2, x1:x2], item.stats[stat_name])

            channels.append(np.clip(channel, 0.0, 1.0))

        return channels

    def _histogram_to_patch(self, histogram: np.ndarray, height: int, width: int) -> np.ndarray:
        if height <= 0 or width <= 0:
            return np.zeros((self.config.hist_bins, 0, 0), dtype=np.float32)

        patch = np.repeat(histogram[:, None], width, axis=1)
        patch = np.repeat(patch[:, None, :], height, axis=1)
        return patch.astype(np.float32)

    def _finalize_feature_stack(
        self,
        accum: np.ndarray,
        counts: np.ndarray,
        already_merged: bool,
    ) -> list[np.ndarray]:
        if already_merged:
            output = accum
        elif self.config.merge_mode == "mean":
            valid = counts > 0
            output = np.zeros_like(accum)
            output[:, valid] = accum[:, valid] / counts[valid]
        else:
            output = accum
        return [np.clip(output[index], 0.0, 1.0) for index in range(output.shape[0])]

    def _discover_dataset_dirs(self, datasets_root: Path, dataset_names: Iterable[str] | None) -> list[Path]:
        dataset_names_set = set(dataset_names or [])

        direct_layout = (datasets_root / "images").is_dir() and (datasets_root / "labels").is_dir()
        if direct_layout:
            return [datasets_root]

        dataset_dirs = []
        for child in sorted(path for path in datasets_root.iterdir() if path.is_dir()):
            if dataset_names_set and child.name not in dataset_names_set:
                continue
            if (child / "images").is_dir() and (child / "labels").is_dir():
                dataset_dirs.append(child)
        return dataset_dirs
