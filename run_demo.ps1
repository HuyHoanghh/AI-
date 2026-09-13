$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$workspacePython = Join-Path (Split-Path $PSScriptRoot -Parent) 'python-runtime/python.exe'
if (-not (Test-Path -LiteralPath $workspacePython)) { $workspacePython = 'python' }
& $workspacePython scripts/export_demo_model.py
if ($LASTEXITCODE -ne 0) { throw 'Không xuất được mô hình demo.' }
& $workspacePython -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
