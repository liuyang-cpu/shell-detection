from __future__ import annotations

import argparse
import json
from pathlib import Path

from histogram_yolo_builder import HistogramYOLOBuilder, HistogramYOLOConfig, VALID_STATS


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build YOLO-ready multi-channel tensors from grayscale X-ray images and YOLO bbox labels."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file")
    parser.add_argument("--datasets-root", type=Path, default=None, help="Root directory containing YOLO datasets")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory for training-ready datasets (images/ labels/ sidecar/ dataset.yaml)",
    )
    parser.add_argument("--background-threshold", type=int, default=250, help="Pixels >= threshold are background")
    parser.add_argument(
        "--foreground-mode",
        choices=("smart", "threshold"),
        default="smart",
        help="Foreground extraction mode: adaptive component selection or legacy fixed threshold",
    )
    parser.add_argument("--hist-bins", type=int, default=16, help="Histogram bin count")
    parser.add_argument(
        "--hist-mode",
        choices=("object", "global"),
        default="object",
        help="Use per-object histogram mapping or one global histogram for the full image",
    )
    parser.add_argument(
        "--merge-mode",
        choices=("mean", "sum", "max"),
        default="mean",
        help="How to merge overlapping objects in feature channels",
    )
    parser.add_argument(
        "--include-stats",
        nargs="*",
        default=["mean", "iqr", "p10", "p90"],
        choices=VALID_STATS,
        help="Optional bbox-level statistics to map into channels",
    )
    parser.add_argument(
        "--save-format",
        choices=("npy", "pt", "both"),
        default="npy",
        help="Tensor save format",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to process")
    parser.add_argument("--dataset-names", nargs="*", default=None, help="Optional subset of child dataset names")
    parser.add_argument(
        "--skip-empty-labels",
        action="store_true",
        help="Skip images whose label files contain no valid YOLO boxes",
    )
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = build_parser()
    if pre_args.config is not None:
        config_data = load_json_config(pre_args.config)
        parser.set_defaults(**config_data)

    args = parser.parse_args()
    if args.datasets_root is None or args.output_root is None:
        parser.error("--datasets-root and --output-root are required, either by CLI or --config")

    config = HistogramYOLOConfig(
        background_threshold=args.background_threshold,
        foreground_mode=args.foreground_mode,
        hist_bins=args.hist_bins,
        hist_mode=args.hist_mode,
        merge_mode=args.merge_mode,
        include_stats=tuple(args.include_stats),
        save_format=args.save_format,
        skip_empty_labels=args.skip_empty_labels,
    )
    builder = HistogramYOLOBuilder(config)
    summary = builder.process_dataset_root(
        datasets_root=Path(args.datasets_root),
        output_root=Path(args.output_root),
        splits=args.splits,
        dataset_names=args.dataset_names,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
