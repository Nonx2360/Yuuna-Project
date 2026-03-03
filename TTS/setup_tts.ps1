$repoUrl = "https://github.com/RVC-Boss/GPT-SoVITS.git"
$destDir = "GPT-SoVITS"

Write-Host "--- Starting GPT-SoVITS Setup ---" -ForegroundColor Cyan

# 1. Clone the repository
if (-not (Test-Path $destDir)) {
    Write-Host "Cloning GPT-SoVITS repository..."
    git clone $repoUrl
} else {
    Write-Host "GPT-SoVITS folder already exists. Skipping clone."
}

# 2. Enter directory
Set-Location $destDir

# 3. Create a virtual environment (optional but recommended)
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# 4. Install requirements
Write-Host "Installing requirements (this may take a while)..."
./venv/Scripts/python -m pip install -r requirements.txt

# 5. Download base models
Write-Host "Downloading pretrained base models..."
./venv/Scripts/python download_models.py

Write-Host "`n--- Setup Complete! ---" -ForegroundColor Green
Write-Host "To start the API: ./venv/Scripts/python api.py"
Write-Host "To start the WebUI: ./venv/Scripts/python webui.py"
