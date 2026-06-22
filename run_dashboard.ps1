# Launch the Crypto Investment Analysis dashboard.
# Usage:  ./run_dashboard.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# First run: if the database is empty, seed offline demo data so the UI isn't blank.
if (-not (Test-Path "data/market.db")) {
    Write-Host "No database found - seeding offline demo data..." -ForegroundColor Yellow
    python -m crypto_tool.cli seed-demo
}

Write-Host "Starting Streamlit dashboard..." -ForegroundColor Green
streamlit run crypto_tool/app/streamlit_app.py
