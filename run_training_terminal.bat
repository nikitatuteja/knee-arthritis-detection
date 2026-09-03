@echo off
title ArthroScan AI - DenseNet-121 Model Training
cd /d "%~dp0"
echo ======================================================================
echo  ARTHROSCAN AI - MODEL TRAINING TERMINAL (Nikita Tuteja)
echo ======================================================================
echo.
call .\venv\Scripts\activate.bat
python train.py --warmup_epochs 3 --finetune_epochs 5
echo.
echo ======================================================================
echo  TRAINING CONCLUDED! Feel free to capture your screenshot now.
echo ======================================================================
pause
