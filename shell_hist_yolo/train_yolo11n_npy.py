from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import YAML


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train yolo11n.pt on a shell histogram NPY dataset exported by build_dataset.py."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to dataset.yaml produced under shell_hist_yolo_output/<dataset_name>/dataset.yaml",
    )
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"), help="Base pretrained YOLO checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Training device, e.g. 0, 0,1 or cpu")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--project", type=Path, default=Path("runs/shell_hist_yolo"), help="Output project folder")
    parser.add_argument("--name", type=str, default="yolo11n_npy", help="Run name")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics image caching")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable AMP training")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load pretrained weights from --model",
    )
    parser.add_argument("--resume", action="store_true", help="Resume the latest matching run")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run directory")
    return parser


def validate_dataset_yaml(data_path: Path) -> dict:
    if not data_path.is_file():
        raise FileNotFoundError(f"dataset yaml not found: {data_path}")

    data = YAML.load(data_path)
    channels = int(data.get("channels", 0) or 0)
    if channels <= 0:
        raise ValueError(
            f"{data_path} is missing a valid 'channels' field. "
            "Rebuild the dataset with shell_hist_yolo/build_dataset.py if needed."
        )
    if "train" not in data or "val" not in data:
        raise ValueError(f"{data_path} must contain both 'train' and 'val' entries")
    return data


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_path = args.data.resolve()
    model_path = args.model.resolve()
    project_path = args.project.resolve()

    data = validate_dataset_yaml(data_path)
    channels = int(data["channels"])
    names = data.get("names", {})
    nc = len(names) if isinstance(names, dict) else len(names or [])

    if not model_path.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {model_path}")

    print(f"Dataset YAML : {data_path}")
    print(f"Model        : {model_path}")
    print(f"Channels     : {channels}")
    print(f"Classes      : {nc}")
    print(f"Project      : {project_path}")

    model = YOLO(model_path)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        patience=args.patience,
        cache=args.cache,
        amp=args.amp,
        pretrained=args.pretrained,
        resume=args.resume,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
