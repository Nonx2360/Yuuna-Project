# Run Yuuna-chan with the project venv
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
& "$ProjectRoot\venv\Scripts\python.exe" app.py
