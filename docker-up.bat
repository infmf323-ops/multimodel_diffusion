@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Project: %CD%
echo Starting: docker compose up --build
echo UI http://localhost:3000  API http://localhost:8000/docs
docker compose up --build
