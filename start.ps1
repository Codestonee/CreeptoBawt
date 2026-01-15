# Quick Start Script for Windows
# Usage: .\start.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Trading Bot - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
$dockerRunning = docker info 2>&1 | Select-String "ERROR"
if ($dockerRunning) {
    Write-Host "ERROR: Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

# Check if .env exists
if (!(Test-Path .env)) {
    Write-Host "ERROR: .env file not found!" -ForegroundColor Red
    Write-Host "Creating from .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "Please edit .env with your API credentials and run again." -ForegroundColor Yellow
    notepad .env
    exit 1
}

# Create directories
New-Item -ItemType Directory -Force -Path data, logs, backups, config, monitoring | Out-Null

# Build images
Write-Host "Building Docker images..." -ForegroundColor Green
docker-compose build --no-cache

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}

# Start services
Write-Host "Starting services..." -ForegroundColor Green
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to start services!" -ForegroundColor Red
    exit 1
}

# Wait for services to be healthy
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check status
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Status Check" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
docker-compose ps

# Show logs
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Recent Logs" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
docker-compose logs --tail 20

Write-Host ""
Write-Host "✅ Bot is running!" -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Logs:      docker-compose logs -f" -ForegroundColor Cyan
Write-Host "Stop:      docker-compose down" -ForegroundColor Cyan
Write-Host ""
