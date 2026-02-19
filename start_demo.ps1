
# Set the current directory to the script's directory
Set-Location $PSScriptRoot

# Define the conda environment path
$CondaEnvPath = "C:\Users\rapha\miniconda3\envs\qwen3-tts-cuda"
$PythonExe = "$CondaEnvPath\python.exe"

# Add conda env binaries to PATH so sox and other tools are found
$env:PATH = "$CondaEnvPath\Library\bin;$CondaEnvPath\Scripts;$CondaEnvPath\bin;$env:PATH"

# Define the model to use
$ModelName = "Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Print status
Write-Host "Starting Qwen3-TTS Demo using environment: qwen3-tts-cuda" -ForegroundColor Green
Write-Host "Model: $ModelName" -ForegroundColor Cyan
Write-Host "Access at: http://127.0.0.1:7860" -ForegroundColor Yellow

# Run the demo (--no-flash-attn because flash-attn is not available on Windows)
& $PythonExe -m qwen_tts.cli.demo $ModelName --ip 127.0.0.1 --port 7860 --no-flash-attn
