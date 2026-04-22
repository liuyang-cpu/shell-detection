param(
    [string]$PythonExe = "python"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$trainScript = Join-Path $scriptDir "train_yolo11n_npy.py"
$configPath = Join-Path $scriptDir "train_config.jsonc"

if (-not (Test-Path $trainScript)) {
    throw "训练脚本不存在: $trainScript"
}

if (-not (Test-Path $configPath)) {
    throw "训练配置不存在: $configPath"
}

Write-Host "训练脚本: $trainScript"
Write-Host "训练配置: $configPath"

& $PythonExe $trainScript --config $configPath
