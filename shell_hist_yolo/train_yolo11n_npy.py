from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
from ultralytics.utils import YAML


def strip_json_comments(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    i = 0
    length = len(text)

    while i < length:
        char = text[i]
        nxt = text[i + 1] if i + 1 < length else ""

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        if char == "/" and nxt == "/":
            i += 2
            while i < length and text[i] not in "\r\n":
                i += 1
            continue

        if char == "/" and nxt == "*":
            i += 2
            while i + 1 < length and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        result.append(char)
        i += 1

    return "".join(result)


def load_json_config(config_path: Path) -> dict[str, object]:
    text = config_path.read_text(encoding="utf-8")
    data = json.loads(strip_json_comments(text))
    if not isinstance(data, dict):
        raise ValueError("config file must contain a JSON object")
    return data


def resolve_path(value: Path | str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train yolo11n.pt on a shell histogram NPY dataset exported by build_dataset.py."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON/JSONC config file. Defaults to shell_hist_yolo/train_config.jsonc when present.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to dataset.yaml produced under shell_hist_yolo_output/<dataset_name>/dataset.yaml",
    )
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"), help="Base pretrained YOLO checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Training device, e.g. cuda, 0, 0,1 or cpu")
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
    # 预处理脚本会把多通道数量写入 dataset.yaml。
    # 训练时必须读取同一个 channels 值，才能按正确输入通道数重建模型。
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
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_config_path = script_dir / "train_config.jsonc"

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = build_parser()
    config_path = pre_args.config or (default_config_path if default_config_path.is_file() else None)
    if config_path is not None:
        parser.set_defaults(**load_json_config(config_path))

    args = parser.parse_args()

    if args.data is None:
        parser.error("--data is required, either by CLI or --config")

    data_path = resolve_path(args.data, repo_root)
    model_path = resolve_path(args.model, repo_root)
    project_path = resolve_path(args.project, repo_root)

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
    train_device = args.device.strip() if isinstance(args.device, str) else str(args.device)
    if not train_device:
        train_device = "cuda"

    print(f"Project      : {project_path}")
    print(f"Device       : {train_device}")

    # 从 yolo11n.pt 加载时，会尽量复用形状匹配的预训练权重。
    # 对于 NPY 多通道输入，Ultralytics 会按 dataset.yaml 里的 channels 重建首层卷积。
    model = YOLO(model_path)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=train_device,
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
