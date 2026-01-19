# Quick status check for preprocessing
$preprocessed = (Get-ChildItem -Path "D:\MA\data\mimic-iv-ecg\icustay_ecgs_24h\preprocessed" -Filter "*.npy" -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
$total = 30000
$progress = [math]::Round($preprocessed / $total * 100, 2)

Write-Host "=" * 50
Write-Host "Preprocessing Status"
Write-Host "=" * 50
Write-Host "Preprocessed ECGs: $preprocessed / $total"
Write-Host "Progress: $progress%"
Write-Host "Remaining: $($total - $preprocessed) ECGs"
Write-Host "=" * 50

