@echo off
setlocal enabledelayedexpansion

REM ===== Settings (customize) =====
set ENC_MODEL=%ENC_MODEL%
if "%ENC_MODEL%"=="" set ENC_MODEL=roberta-base
set DEC_MODEL=%DEC_MODEL%
if "%DEC_MODEL%"=="" set DEC_MODEL=meta-llama/Llama-3.2-3B
set EMBED_MODEL=%EMBED_MODEL%
if "%EMBED_MODEL%"=="" set EMBED_MODEL=BAAI/bge-small-en-v1.5
set TOPK=%TOPK%
if "%TOPK%"=="" set TOPK=4
set K=%K%
if "%K%"=="" set K=32
set P=%P%
if "%P%"=="" set P=0.25
set CTX_MAX=%CTX_MAX%
if "%CTX_MAX%"=="" set CTX_MAX=1024
set MAX_NEW=%MAX_NEW%
if "%MAX_NEW%"=="" set MAX_NEW=128
set STEPS=%STEPS%
if "%STEPS%"=="" set STEPS=200
set LR_RECON=%LR_RECON%
if "%LR_RECON%"=="" set LR_RECON=2e-5
set LR_NEXT=%LR_NEXT%
if "%LR_NEXT%"=="" set LR_NEXT=2e-5
set LR_POLICY=%LR_POLICY%
if "%LR_POLICY%"=="" set LR_POLICY=1e-4
REM ================================

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

if not exist "refrag.py" (
  echo ERROR: refrag.py not found in %SCRIPT_DIR%. Place refrag.py next to this script.
  exit /b 1
)

REM ---- Python & venv ----
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Please install Python 3.10+ and add it to PATH.
  exit /b 1
)

python -m venv .venv
call .venv\Scripts\activate

python -m pip install --upgrade pip

REM ---- Detect NVIDIA and install Torch accordingly ----
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo Installing PyTorch CUDA (cu121)
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
  echo Installing CPU-only PyTorch
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

REM FAISS on Windows: use CPU (faiss-gpu wheels are not provided for Windows on pip)
pip install faiss-cpu

REM Common deps
pip install "transformers==4.43.3" accelerate sentencepiece sacrebleu numpy

REM ---- Patch refrag.py to use MPS (ignored on Windows), and prefer CUDA when available ----
python - <<PY
import io,re,sys,pathlib
p=pathlib.Path("refrag.py")
s=p.read_text(encoding="utf-8")
if "torch.backends.mps" not in s:
    s=re.sub(
        r"def now_device\(\):\n[^\n]*return torch\.device\('[^']+'\)[^\n]*\n",
        "def now_device():\n"
        "    if hasattr(torch, 'cuda') and torch.cuda.is_available():\n"
        "        return torch.device('cuda')\n"
        "    if hasattr(torch.backends, 'mps') and getattr(torch.backends, 'mps').is_available():\n"
        "        return torch.device('mps')\n"
        "    return torch.device('cpu')\n",
        s, flags=re.DOTALL)
    p.write_text(s, encoding="utf-8")
    print("[patch] Updated now_device() to support CUDA → MPS → CPU.")
else:
    print("[patch] MPS logic already present; no change.")
PY

set TOKENIZERS_PARALLELISM=false

REM 1) Toy corpus + index
mkdir data 2>nul
mkdir runs\index 2>nul

(
echo Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London.
echo The capital of France is Paris.
echo Alan Turing proposed the Turing test in 1950.
echo Penicillin is an antibiotic derived from Penicillium fungi.
echo Large language models can use retrieval to augment their context.
) > data\wiki_lines.txt

python refrag.py index ^
  --corpus data\wiki_lines.txt ^
  --index_dir runs\index ^
  --embed_model %EMBED_MODEL%

REM 2) Quick generate
python refrag.py generate ^
  --index_dir runs\index ^
  --embed_model %EMBED_MODEL% ^
  --enc %ENC_MODEL% ^
  --dec %DEC_MODEL% ^
  --question "Who discovered penicillin?" ^
  --topk %TOPK% ^
  --k %K% ^
  --p %P% ^
  --ctx_max %CTX_MAX% ^
  --max_new %MAX_NEW% ^
  --temperature 0.0

REM 3) CPT datasets
(
echo {"id":"ex1","tokens":"Penicillin revolutionized medicine by enabling treatment of bacterial infections.","split":{"s":1024,"o":128}}
echo {"id":"ex2","tokens":"Alan Turing's work laid the foundations of computer science and artificial intelligence.","split":{"s":1024,"o":128}}
echo {"id":"ex3","tokens":"Paris is the capital and most populous city of France, known for art, fashion, and gastronomy.","split":{"s":1024,"o":128}}
) > data\cpt_train.jsonl

REM 3A) Reconstruction
python refrag.py cpt_recon ^
  --train_json data\cpt_train.jsonl ^
  --enc %ENC_MODEL% ^
  --dec %DEC_MODEL% ^
  --k 64 ^
  --steps %STEPS% ^
  --lr %LR_RECON% ^
  --log_every 20 ^
  --out_dir runs\cpt_recon

REM 3B) Next-paragraph
python refrag.py cpt_next ^
  --train_json data\cpt_train.jsonl ^
  --enc %ENC_MODEL% ^
  --dec %DEC_MODEL% ^
  --k 64 ^
  --steps %STEPS% ^
  --lr %LR_NEXT% ^
  --expand_frac 0.25 ^
  --log_every 20 ^
  --load_dir runs\cpt_recon ^
  --out_dir runs\cpt_next

REM 4) Policy training
(
echo {"id":"q1","question":"Who discovered penicillin?","answers":["Alexander Fleming"]}
echo {"id":"q2","question":"What is the capital of France?","answers":["Paris"]}
) > data\rag_train.jsonl

python refrag.py train_policy ^
  --rag_json data\rag_train.jsonl ^
  --index_dir runs\index ^
  --embed_model %EMBED_MODEL% ^
  --enc %ENC_MODEL% ^
  --dec %DEC_MODEL% ^
  --k 32 ^
  --steps %STEPS% ^
  --lr %LR_POLICY% ^
  --p %P% ^
  --topk %TOPK% ^
  --log_every 20 ^
  --out_dir runs\policy

echo ---- Generate with trained policy ----
python refrag.py generate ^
  --index_dir runs\index ^
  --embed_model %EMBED_MODEL% ^
  --enc %ENC_MODEL% ^
  --dec %DEC_MODEL% ^
  --load_dir runs\policy ^
  --question "Explain how penicillin was discovered and by whom." ^
  --topk %TOPK% --k %K% --p %P% --max_new 192

echo ---- Generate with CPT-tuned full model ----
python refrag.py generate ^
  --index_dir runs\index ^
  --embed_model %EMBED_MODEL% ^
  --enc %ENC_MODEL% ^
  --dec %DEC_MODEL% ^
  --load_dir runs\cpt_next ^
  --question "Explain how penicillin was discovered and by whom." ^
  --topk %TOPK% --k %K% --p %P% --max_new 192

echo ✅ Done.
