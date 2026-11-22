$ErrorActionPreference = "Stop"
$root = Join-Path $env:LOCALAPPDATA "Roadframe"
New-Item -ItemType Directory -Force $root | Out-Null

function Get-VerifiedFile($url, $name, $sha) {
    $dest = Join-Path $root $name
    if ((Test-Path -LiteralPath $dest) -and (Get-FileHash -LiteralPath $dest).Hash -eq $sha) { return }
    $temp = "$dest.download"
    Write-Host "Downloading $name"
    curl.exe --fail --location --retry 2 --output $temp $url
    if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath $temp).Hash -ne $sha) {
        throw "Download or checksum failed for $name"
    }
    Move-Item -LiteralPath $temp -Destination $dest -Force
}

$release = "https://github.com/ggml-org/llama.cpp/releases/download/b12628"
$weights = "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/1cd86afb9a95c410a6038ab3b40d8b578c892266"
Get-VerifiedFile "$release/llama-b12628-bin-win-vulkan-x64.zip" "runtime.zip" "dd28b411feb15b2e2b8bcf6bdb182a33962f27509d0f0e334a6604b11a00cf9a"
Get-VerifiedFile "$weights/Qwen3VL-4B-Instruct-Q4_K_M.gguf" "Qwen3VL-4B-Instruct-Q4_K_M.gguf" "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a"
Get-VerifiedFile "$weights/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf" "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf" "30ba2c7dd3127a4561b6cba9d13d0f711c91bdb38742e2f56d73c8cb596bd06d"
Expand-Archive -LiteralPath (Join-Path $root "runtime.zip") -DestinationPath (Join-Path $root "runtime") -Force
Write-Host "Model ready. Start server.py and select the Russian traffic analysis option."
