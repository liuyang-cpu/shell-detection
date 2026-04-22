# Shell Histogram YOLO 预处理说明

这个目录提供了一个独立的工业 X 光贝壳缺陷预处理流程。

它会读取灰度图和对应的 YOLO 标注，在每个 bbox 内提取前景相关的灰度特征，并生成保持原始空间尺寸不变的多通道张量。

## 生成内容

每张图最终会生成一个形状为：

```text
(C, H, W)
```

的张量，通道组成如下：

- 通道 0：归一化后的灰度图
- 通道 1..N：直方图特征通道
- 可选附加通道：`mean`、`iqr`、`p10`、`p90`、`foreground_ratio` 等统计量

默认输出格式是 `.npy`。如果环境安装了 `torch`，也支持输出 `.pt`。

## 支持的数据结构

### 结构 1：单个 YOLO 数据集

```text
datasets/
  images/
    train/
    val/
  labels/
    train/
    val/
```

检测到这种结构时，输出会写到 `root_dataset/` 目录下，并整理成训练侧友好的目录结构。

### 结构 2：一个根目录下包含多个子数据集

```text
datasets/
  baibei/
    images/train
    images/val
    labels/train
    labels/val
  baiha/
    ...
```

脚本会分别处理每个符合结构的子数据集。

## 默认行为

- 前景提取模式：`smart`
- 回退阈值：`250`
- 直方图 bin 数：`16`
- 直方图模式：`object`
- 重叠区域合并方式：`mean`
- 额外统计通道：`mean`、`iqr`、`p10`、`p90`
- 保存格式：`npy`

### 前景提取

- `smart`：在 bbox 内先做 Otsu 阈值分割，再选择主连通域
- `threshold`：旧逻辑，直接将 `< background_threshold` 的像素视为前景

如果 `smart` 没找到可用前景，代码会回退到 `threshold` 规则。

### 直方图映射

- `object`：每个 bbox 单独计算归一化直方图，只映射回自己的 bbox 区域
- `global`：把所有对象的直方图合并成整图级直方图，再广播到整张图

### 重叠区域处理

当多个 bbox 有重叠时，特征通道支持：

- `mean`
- `sum`
- `max`

## 快速开始

在项目根目录运行：

```powershell
python shell_hist_yolo\build_dataset.py `
  --datasets-root datasets `
  --output-root shell_hist_yolo_output
```

只处理一个子数据集和一个 split：

```powershell
python shell_hist_yolo\build_dataset.py `
  --datasets-root datasets `
  --output-root shell_hist_yolo_output `
  --dataset-names baibei `
  --splits val
```

使用全局直方图模式：

```powershell
python shell_hist_yolo\build_dataset.py `
  --datasets-root datasets `
  --output-root shell_hist_yolo_output_global `
  --hist-mode global
```

通过 JSONC 配置文件运行：

```powershell
python shell_hist_yolo\build_dataset.py `
  --config shell_hist_yolo\config.example.jsonc
```

## 主要参数

- `--config`：可选，支持 JSON 或 JSONC 配置文件
- `--datasets-root`：数据集根目录
- `--output-root`：张量和元数据输出目录
- `--background-threshold`：旧阈值模式下的背景阈值，合法范围 `1..255`
- `--foreground-mode`：`smart` 或 `threshold`
- `--hist-bins`：直方图 bin 数，必须大于 `0`
- `--hist-mode`：`object` 或 `global`
- `--merge-mode`：`mean`、`sum` 或 `max`
- `--include-stats`：可选子集 `mean iqr p10 p90 foreground_ratio`
- `--save-format`：`npy`、`pt` 或 `both`
- `--splits`：要处理的划分，默认 `train val`
- `--dataset-names`：只处理指定的子数据集
- `--skip-empty-labels`：跳过没有有效 YOLO 框的图片

`--datasets-root` 和 `--output-root` 必须提供，可以来自命令行，也可以来自 `--config`。

## 输出结构

示例：

```text
shell_hist_yolo_output/
  metadata.json
  baibei/
    train/
      image_001.npy
      image_001.json
    val/
```

说明：

当前版本会把输出整理成 Ultralytics 更容易接入的布局：

```text
shell_hist_yolo_output/
  metadata.json
  baibei/
    dataset.yaml
    dataset_metadata.json
    images/
      train/
        image_001.npy
      val/
    labels/
      train/
        image_001.txt
      val/
    sidecar/
      train/
        image_001.json
      val/
```

说明：

- `images/train|val`：模型输入张量，默认是 `.npy`
- `labels/train|val`：复制出的同名 YOLO 标签文件
- `sidecar/train|val`：单张图片的 sidecar 元数据
- `dataset.yaml`：训练侧数据集描述文件，可直接给 Ultralytics 使用
- `dataset_metadata.json`：单个数据集级别的预处理配置与通道信息
- 根目录 `metadata.json`：整次批处理的汇总和实际配置

每图 sidecar `.json` 包含：

- 图像路径
- 标签路径
- 张量形状
- 图像宽高
- 对象数量
- `hist_bins`、`hist_mode`、`merge_mode`
- `background_threshold`、`foreground_mode`
- 启用的统计通道
- 每个对象的 `class_id`、`bbox_xyxy`、`foreground_count` 和统计值

## 通道数计算

总通道数公式为：

```text
1 + hist_bins + len(include_stats)
```

例如：

- `hist_bins = 16`
- `include_stats = ["mean", "iqr", "p10", "p90"]`

则输出通道数为：

```text
1 + 16 + 4 = 21
```

## 接入训练侧的注意事项

- YOLO 标签文件保持不变，现在会自动复制到输出目录下的 `labels/`
- 训练代码需要改为读取 `.npy` 或 `.pt`，而不是原始图片
- 如果通道数太多，可以减小 `hist_bins` 或减少统计通道
- `pt` 输出依赖 `torch`，`npy` 不依赖
- `dataset.yaml` 会自动生成，但其中的 `names` 默认写成 `class_0`、`class_1` 这类占位名，训练前最好改成真实类别名
- `dataset.yaml` 底部会额外写入注释形式的 `input_channels`，方便你同步修改 YOLO 模型的输入通道数

## 目录文件说明

- `histogram_yolo_builder.py`：核心预处理逻辑
- `build_dataset.py`：命令行入口
- `config.example.jsonc`：带注释的示例配置
- `FOREGROUND_EXTRACTION_zh-CN.md`：前景提取说明
- `USAGE_zh-CN.md`：更完整的中文使用文档
