param(
    [string]$DatasetName = "baibei",
    [string]$ModelPath = "yolo11n.pt",
    [string]$Device = "0",
    [int]$Epochs = 100,
    [int]$Imgsz = 640,
    [int]$Batch = 8,
    [int]$Workers = 4,
    [string]$PythonExe = "python"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$dataYaml = Join-Path $repoRoot "shell_hist_yolo_output\$DatasetName\dataset.yaml"
$trainScript = Join-Path $scriptDir "train_yolo11n_npy.py"
$resolvedModelPath = if ([System.IO.Path]::IsPathRooted($ModelPath)) { $ModelPath } else { Join-Path $repoRoot $ModelPath }

if (-not (Test-Path $trainScript)) {
    throw "训练脚本不存在: $trainScript"
}

if (-not (Test-Path $dataYaml)) {
    throw "数据集配置不存在: $dataYaml"
}

if (-not (Test-Path $resolvedModelPath)) {
    throw "模型文件不存在: $resolvedModelPath"
}

Write-Host "训练脚本: $trainScript"
Write-Host "数据配置: $dataYaml"
Write-Host "模型路径: $resolvedModelPath"
Write-Host "训练设备: $Device"

& $PythonExe $trainScript `
    --data $dataYaml `
    --model $resolvedModelPath `
    --epochs $Epochs `
    --imgsz $Imgsz `
    --batch $Batch `
    --device $Device `
    --workers $Workers
