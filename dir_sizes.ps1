$dirs = Get-ChildItem -Path "c:\Users\RohithSuryaCKM\Downloads\Projects\Image_detection" -Directory
foreach ($d in $dirs) {
    $size = (Get-ChildItem $d.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $sizeMB = [math]::Round($size / 1MB, 1)
    Write-Output "$($d.Name): $sizeMB MB"
}
