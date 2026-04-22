# Shell Histogram YOLO 预处理使用说明

## 1. 工具作用

这个工具把工业 X 光灰度图和对应的 YOLO 标注，转换成可供训练侧读取的多通道张量。

输出保持原始图像空间尺寸不变，形状为：

```text
(C, H, W)
```

通道组成如下：

- 通道 0：归一化灰度图，范围 `0~1`
- 通道 1..N：直方图特征通道
- 可选附加通道：`mean`、`iqr`、`p10`、`p90`、`foreground_ratio`

标签文件不会被修改。

## 2. 支持的数据结构

### 结构 A：单个 YOLO 数据集

```text
datasets/
  images/
    train/
    val/
  labels/
    train/
    val/
```

如果是这种结构，输出目录下会使用 `root_dataset/` 作为数据集名。

### 结构 B：一个根目录下包含多个子数据集

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

脚本会自动遍历所有符合结构的子目录，也可以通过 `--dataset-names` 只处理其中几个。

## 3. 实际处理流程

每张图的处理逻辑是：

1. 读取灰度图
2. 读取同名 YOLO 标签
3. 把 YOLO 标注从归一化坐标转换成像素坐标
4. 对每个 bbox 裁剪局部区域
5. 提取前景像素
6. 根据前景像素计算归一化直方图和统计量
7. 把特征映射回整张图对应区域
8. 按配置合并重叠区域
9. 保存张量和 sidecar 元数据

如果 `--skip-empty-labels` 打开，空标签或无有效框的图片会被跳过。

## 4. 默认配置

当前代码默认值如下：

- `foreground_mode = smart`
- `background_threshold = 250`
- `hist_bins = 16`
- `hist_mode = object`
- `merge_mode = mean`
- `include_stats = mean iqr p10 p90`
- `save_format = npy`
- `splits = train val`
- `dataset_names = null`
- `skip_empty_labels = false`

## 5. 前景提取模式

### `smart`

默认模式。

代码会先在 bbox 内做 Otsu 阈值分割，再从候选区域中选择一个主连通域作为前景。选择时会同时考虑连通域面积和其与 bbox 中心的距离。

如果这个流程没有得到可用前景，代码会回退到 `threshold` 规则。

### `threshold`

旧逻辑。

规则是：

```text
pixel < background_threshold -> 前景
pixel >= background_threshold -> 背景
```

因此 `background_threshold` 只在这个模式里直接生效。

## 6. 两种直方图模式

### `object`

推荐默认使用。

每个 bbox 单独计算自己的直方图，然后只广播回自己的 bbox 区域。bbox 外保持为 0。

更适合多目标检测场景，因为每个目标保留自己的灰度分布。

### `global`

代码会把图中所有对象的直方图按前景像素数量加权合并成一个全局直方图，然后把这个全局直方图广播到整张图。

这意味着 `global` 模式下，整张图共享同一组直方图通道，而不是每个 bbox 一套。

## 7. 重叠区域如何合并

当多个 bbox 覆盖到同一像素时，支持三种合并方式：

- `mean`
- `sum`
- `max`

说明：

- 直方图通道支持 `mean / sum / max`
- 统计通道也支持这三种方式
- 最终所有特征通道都会被裁剪到 `0~1`

## 8. 快速使用

在项目根目录运行：

```powershell
python shell_hist_yolo\build_dataset.py `
  --datasets-root datasets `
  --output-root shell_hist_yolo_output
```

这条命令会使用默认参数处理 `train` 和 `val`。

## 9. 常用命令示例

### 示例 1：只处理一个子数据集

```powershell
python shell_hist_yolo\build_dataset.py `
  --datasets-root datasets `
  --output-root shell_hist_yolo_output `
  --dataset-names baibei
```

### 示例 2：只处理 `val`

```powershell
python shell_hist_yolo\build_dataset.py `
  --datasets-root datasets `
  --output-root shell_hist_yolo_output `
  --splits val
```

### 示例 3：使用全局直方图

```powershell
python shell_hist_yolo\build_dataset.py `
  --datasets-root datasets `
  --output-root shell_hist_yolo_output_global `
  --hist-mode global
```

### 示例 4：显式使用旧阈值前景规则

```powershell
python shell_hist_yolo\build_dataset.py `
  --datasets-root datasets `
  --output-root shell_hist_yolo_output_threshold `
  --foreground-mode threshold `
  --background-threshold 245
```

### 示例 5：减少输出通道数

```powershell
python shell_hist_yolo\build_dataset.py `
  --datasets-root datasets `
  --output-root shell_hist_yolo_output_small `
  --hist-bins 4 `
  --include-stats mean
```

此时总通道数是：

```text
1 + 4 + 1 = 6
```

### 示例 6：通过配置文件运行

```powershell
python shell_hist_yolo\build_dataset.py `
  --config shell_hist_yolo\build_config.jsonc
```

`build_config.jsonc` 是带注释的默认配置，脚本会先去掉注释再按 JSON 解析；现在直接运行 `build_dataset.py` 也会默认读取它。

## 10. 参数说明

### `--config`

可选配置文件，支持：

- `.json`
- `.jsonc`

命令行参数会在解析阶段生效；如果同时传入 `--config`，配置文件内容会先作为默认值载入。

### `--datasets-root`

数据集根目录。可以是单个 YOLO 数据集，也可以是包含多个子数据集的目录。

### `--output-root`

输出目录。会在这里生成训练侧友好的数据集结构，包括：

- `images/train|val`
- `labels/train|val`
- `sidecar/train|val`
- `dataset.yaml`
- `dataset_metadata.json`
- 根目录 `metadata.json`

### `--background-threshold`

整数范围 `1..255`。

在 `threshold` 模式下，像素值大于等于该阈值时视为背景。

### `--foreground-mode`

可选值：

- `smart`
- `threshold`

推荐先用 `smart`。

### `--hist-bins`

直方图桶数，必须大于 `0`。

常见可选值：

- `4`
- `8`
- `16`
- `32`

### `--hist-mode`

可选值：

- `object`
- `global`

### `--merge-mode`

可选值：

- `mean`
- `sum`
- `max`

### `--include-stats`

可选统计通道：

- `mean`
- `iqr`
- `p10`
- `p90`
- `foreground_ratio`

例如：

```powershell
--include-stats mean iqr p10 p90
```

如果不传，默认就是 `mean iqr p10 p90`。

### `--save-format`

可选值：

- `npy`
- `pt`
- `both`

说明：

- `npy` 只依赖 `numpy`
- `pt` 和 `both` 需要安装 `torch`

### `--splits`

指定要处理的数据划分，例如：

```powershell
--splits train val
```

或：

```powershell
--splits val
```

### `--dataset-names`

只处理指定的子数据集，例如：

```powershell
--dataset-names baibei baiha
```

### `--skip-empty-labels`

跳过空标签或不包含有效 YOLO 框的图片。

## 11. 输出目录与元数据

输出目录示例：

```text
shell_hist_yolo_output/
  metadata.json
  baibei/
    dataset.yaml
    dataset_metadata.json
    images/
      train/
        图1.npy
      val/
    labels/
      train/
        图1.txt
      val/
    sidecar/
      train/
        图1.json
      val/
```

文件说明：

- `images/*.npy` / `images/*.pt`：多通道张量
- `labels/*.txt`：复制出的同名 YOLO 标签
- `sidecar/*.json`：该图片的 sidecar 元数据
- `dataset.yaml`：给 Ultralytics 使用的数据集配置
- `dataset_metadata.json`：单个数据集的通道数和预处理参数摘要
- 根目录 `metadata.json`：本次运行的汇总信息

每图 sidecar 当前包含这些字段：

- `image_path`
- `label_path`
- `shape`
- `height`
- `width`
- `object_count`
- `hist_bins`
- `hist_mode`
- `merge_mode`
- `background_threshold`
- `foreground_mode`
- `include_stats`
- `objects`

其中 `objects` 里每个元素包含：

- `class_id`
- `bbox_xyxy`
- `foreground_count`
- `stats`

`sidecar/*.json` 主要用于调试、质检和追溯，不是 YOLO 训练主流程的必需输入。

## 12. 通道数计算

总通道数公式：

```text
1 + hist_bins + len(include_stats)
```

例如：

- `hist_bins = 16`
- `include_stats = ["mean", "iqr", "p10", "p90"]`

则输出通道数是：

```text
1 + 16 + 4 = 21
```

## 13. 如何读取结果

### NumPy 读取

```python
import numpy as np

x = np.load("shell_hist_yolo_output/baibei/images/train/图1.npy")
print(x.shape)  # (C, H, W)
print(x.dtype)  # float32
```

### PyTorch 读取

```python
import numpy as np
import torch

x = np.load("shell_hist_yolo_output/baibei/images/train/图1.npy").astype("float32")
x = torch.from_numpy(x)
print(x.shape)
```

## 14. 接入训练侧时要改什么

这个工具会把 YOLO 标签复制到输出目录的 `labels/` 下，所以训练侧一般只需要调整图像读取逻辑：

1. 保留原有标签解析逻辑
2. 把原来读取 `.png/.jpg` 的部分改成读取 `images/` 下的 `.npy` 或 `.pt`
3. 模型 YAML 的输入通道数 `ch` 要改成当前特征通道数
4. `dataset.yaml` 可以直接作为数据集配置起点使用

也就是说，数据集类需要从“读取普通图像”改成“读取 `(C, H, W)` 张量”。

## 15. 注意事项

- 如果通道数太多，先减小 `hist_bins` 或减少 `include_stats`
- 如果必须保持接近 3 通道输入，需要在配置上主动压缩通道数
- 如果选择 `pt` 或 `both`，运行环境里必须有 `torch`
- 缺少同名标签文件的图片会被直接跳过
- 标签行格式不是标准 YOLO `class cx cy w h` 的记录也会被跳过

## 16. 相关文件

- 核心代码：`shell_hist_yolo/histogram_yolo_builder.py`
- 命令入口：`shell_hist_yolo/build_dataset.py`
- 示例配置：`shell_hist_yolo/build_config.jsonc`
- 英文说明：`shell_hist_yolo/README.md`
