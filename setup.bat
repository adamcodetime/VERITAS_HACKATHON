@echo off

if not exist venv (
    uv venv venv
)

call venv\Scripts\activate

uv pip install ^
torch ^
pandas ^
numpy ^
kagglehub ^
yfinance ^
onnx ^
onnxruntime ^
onnxscript

python train.py

pause