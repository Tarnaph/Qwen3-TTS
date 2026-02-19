
# Set the current directory to the script's directory
Set-Location $PSScriptRoot

# Define the conda environment path
$CondaEnvPath = "C:\Users\rapha\miniconda3\envs\qwen3-tts-cuda"
$PythonExe = "$CondaEnvPath\python.exe"

# Add conda env binaries to PATH so sox and other tools are found
$env:PATH = "$CondaEnvPath\Library\bin;$CondaEnvPath\Scripts;$CondaEnvPath\bin;$env:PATH"

# Base model path (supports voice cloning)
$ModelName = "Qwen3-TTS-12Hz-1.7B-Base"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Qwen3-TTS - Voice Clone Demo" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Model: $ModelName" -ForegroundColor Green
Write-Host "Access at: http://127.0.0.1:7861" -ForegroundColor Yellow
Write-Host ""
Write-Host "How to use:" -ForegroundColor White
Write-Host "  1. Upload a reference audio of YOUR voice (5-15 seconds)" -ForegroundColor Gray
Write-Host "  2. Type the reference text (what was said in the audio)" -ForegroundColor Gray
Write-Host "  3. Type the text you want to synthesize" -ForegroundColor Gray
Write-Host "  4. Click Generate" -ForegroundColor Gray
Write-Host ""

# Run the demo with the Base model on port 7861 (--no-flash-attn for Windows compatibility)
& $PythonExe -m qwen_tts.cli.demo $ModelName --ip 127.0.0.1 --port 7861 --no-flash-attn
