
#!/bin/bash
# scripts/setup_runpod.sh
# =======================
# Setup de entorno en RunPod / vast.ai para entrenar el Santiago World Model
# GPU recomendada: A100 80GB (más cómodo) o A100 40GB (ajustado)
# Costo estimado RunPod: ~$2.5/h × 4h = ~$10 USD por run completo
#
# Uso:
#   chmod +x scripts/setup_runpod.sh
#   ./scripts/setup_runpod.sh

set -e

echo "╔══════════════════════════════════════════════╗"
echo "║  Santiago World Model — RunPod Setup          ║"
echo "║  VibeCodingChile                              ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. Verificar GPU ──────────────────────────────────────────────
echo "[1/8] Verificando GPU..."
nvidia-smi
echo ""
python3 -c "
import torch
props = torch.cuda.get_device_properties(0)
total_gb = props.total_memory / 1e9
print(f'GPU: {props.name}')
print(f'VRAM: {total_gb:.1f} GB')
if total_gb >= 79:
    print('Config: A100 80GB → rank=128 posible')
elif total_gb >= 39:
    print('Config: A100 40GB → rank=64 recomendado')
elif total_gb >= 23:
    print('Config: RTX 4090/3090 → rank=32 recomendado')
else:
    print('⚠ VRAM baja → rank=16 máximo')
"

# ── 2. Sistema ────────────────────────────────────────────────────
echo "[2/8] Actualizando sistema..."
apt-get update -qq
apt-get install -y -qq git wget curl ffmpeg libsm6 libxext6

# ── 3. Python deps ────────────────────────────────────────────────
echo "[3/8] Instalando dependencias Python..."
pip install --upgrade pip -q

# Core
pip install -q \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# xformers para memoria más eficiente
pip install -q xformers --index-url https://download.pytorch.org/whl/cu121

# Diffusers y PEFT
pip install -q \
    diffusers>=0.30.0 \
    transformers>=4.44.0 \
    accelerate>=0.30.0 \
    peft>=0.11.0 \
    huggingface_hub>=0.23.0

# Data pipeline
pip install -q \
    scikit-learn \
    Pillow \
    requests \
    tqdm \
    pyyaml \
    numpy \
    imageio \
    imageio-ffmpeg \
    opencv-python-headless

echo "✓ Dependencias instaladas"

# ── 4. Repo ───────────────────────────────────────────────────────
echo "[4/8] Clonando repositorio..."
if [ ! -d "santiago-world-model" ]; then
    git clone https://github.com/vibecodingchile/santiago-world-model.git
fi
cd santiago-world-model

# ── 5. HuggingFace login ──────────────────────────────────────────
echo "[5/8] HuggingFace login..."
echo "Ingresa tu token de HuggingFace (o presiona Enter para usar CogVideoX):"
read -r HF_TOKEN

if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
    MODEL_ID="nvidia/Cosmos-Predict2.5-2B-Video2World"
    echo "✓ Login exitoso — usando Cosmos"
else
    MODEL_ID="THUDM/CogVideoX-5b"
    echo "ℹ Sin token — usando CogVideoX-5b (mismo DiT, sin licencia)"
fi

export SWM_MODEL_ID=$MODEL_ID

# ── 6. Mapillary ──────────────────────────────────────────────────
echo "[6/8] Token Mapillary para datos reales (opcional):"
read -r MAPILLARY_TOKEN

if [ -n "$MAPILLARY_TOKEN" ]; then
    export MAPILLARY_TOKEN=$MAPILLARY_TOKEN
    echo "Descargando datos de Santiago..."
    python3 -m data_pipeline.mapillary_scraper \
        --token "$MAPILLARY_TOKEN" \
        --region santiago_centro \
        --limit 5000 \
        --resolution 1024 \
        --output ./data/mapillary \
        --workers 16
    echo "✓ Datos descargados"
else
    echo "ℹ Sin token Mapillary — se usarán datos sintéticos"
fi

# ── 7. Configurar VRAM ────────────────────────────────────────────
echo "[7/8] Optimizando configuración para GPU disponible..."

VRAM_GB=$(python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1e9))")

if [ "$VRAM_GB" -ge 79 ]; then
    LORA_RANK=128
    IMAGE_SIZE=512
    echo "A100 80GB detectado: rank=128, size=512"
elif [ "$VRAM_GB" -ge 39 ]; then
    LORA_RANK=64
    IMAGE_SIZE=480
    echo "A100 40GB detectado: rank=64, size=480"
elif [ "$VRAM_GB" -ge 23 ]; then
    LORA_RANK=32
    IMAGE_SIZE=320
    echo "24GB GPU detectado: rank=32, size=320"
else
    LORA_RANK=16
    IMAGE_SIZE=256
    echo "GPU pequeña detectada: rank=16, size=256"
fi

# Variables de entorno para PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export XFORMERS_ENABLED=1

# ── 8. Lanzar entrenamiento ───────────────────────────────────────
echo "[8/8] Iniciando entrenamiento..."
echo ""
echo "Configuración final:"
echo "  Modelo:    $MODEL_ID"
echo "  LoRA rank: $LORA_RANK"
echo "  Img size:  ${IMAGE_SIZE}px"
echo "  Steps:     3000"
echo ""

python3 train_lora.py \
    --max-steps 3000 \
    --rank "$LORA_RANK" \
    --dtype bf16 \
    --output-dir ./checkpoints/swm_scl_lora

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✓ Entrenamiento completado                   ║"
echo "║  Checkpoints: ./checkpoints/swm_scl_lora      ║"
echo "╚══════════════════════════════════════════════╝"

# Copiar pesos LoRA (los únicos que necesitas guardar)
echo ""
echo "Para descargar solo los pesos LoRA (~200-400MB):"
echo "  scp -r ./checkpoints/swm_scl_lora/final/lora_weights usuario@tu-maquina:~/"
