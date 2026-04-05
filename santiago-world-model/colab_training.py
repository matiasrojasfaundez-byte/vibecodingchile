# ╔══════════════════════════════════════════════════════════════════════════╗
# ║        SANTIAGO WORLD MODEL — Google Colab Fine-tune Notebook           ║
# ║        VibeCodingChile × Seoul World Model (NAVER AI Lab / KAIST)       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# Instrucciones:
# 1. Subir este archivo a Google Colab (o copiar celda por celda)
# 2. Runtime → Change runtime type → A100 GPU (Colab Pro+)
# 3. Correr todas las celdas en orden
#
# VRAM estimada: ~32-38GB de 40GB (A100)
# Tiempo estimado: ~3-4h para 3000 pasos con dataset sintético


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 1: Verificar GPU                                                  │
# └─────────────────────────────────────────────────────────────────────────┘

import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")

# Esperado en Colab Pro+:
# GPU: NVIDIA A100-SXM4-40GB
# VRAM: 40.0 GB


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 2: Instalar dependencias                                          │
# └─────────────────────────────────────────────────────────────────────────┘

# Core ML
# !pip install -q diffusers>=0.30.0 transformers>=4.44.0 accelerate>=0.30.0
# !pip install -q peft>=0.11.0
# !pip install -q xformers --index-url https://download.pytorch.org/whl/cu121

# Data pipeline
# !pip install -q scikit-learn Pillow requests tqdm pyyaml

# Para Mapillary (opcional — si vas a scraping real)
# !pip install -q mapillary

# Video output
# !pip install -q imageio imageio-ffmpeg

print("✓ Dependencias instaladas")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 3: HuggingFace login (para Cosmos — requiere licencia aceptada)  │
# └─────────────────────────────────────────────────────────────────────────┘

# Opción A: Cosmos (más potente, requiere licencia NVIDIA)
# 1. Ir a: https://huggingface.co/nvidia/Cosmos-Predict2.5-2B-Video2World
# 2. Aceptar los términos de uso
# 3. Crear token en: https://huggingface.co/settings/tokens

# from huggingface_hub import login
# login(token="hf_TU_TOKEN_AQUI")

# Opción B: CogVideoX-5b (sin licencia especial, misma arquitectura DiT)
# Descomentar la línea de model_id abajo para usar este.

MODEL_ID = "nvidia/Cosmos-Predict2.5-2B-Video2World"
# MODEL_ID = "THUDM/CogVideoX-5b"   # ← usar este si no tienes licencia Cosmos

print(f"Modelo seleccionado: {MODEL_ID}")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 4: Clonar repo y configurar                                       │
# └─────────────────────────────────────────────────────────────────────────┘

import os
import sys

# Clonar el repo (cuando esté en GitHub)
# !git clone https://github.com/vibecodingchile/santiago-world-model
# %cd santiago-world-model

# Por ahora: usar los archivos directamente
# Sube train_lora.py y la carpeta data_pipeline/ a Colab

# Verificar estructura
# !ls -la

# Configurar variables de entorno
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Mapillary token (si vas a scraping real)
# os.environ["MAPILLARY_TOKEN"] = "tu_token_aqui"

print("✓ Entorno configurado")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 5: (Opcional) Scraping de datos reales de Santiago               │
# └─────────────────────────────────────────────────────────────────────────┘

# Si tienes token de Mapillary, descarga datos reales:
#
# from data_pipeline.mapillary_scraper import MapillaryScraper
#
# scraper = MapillaryScraper(
#     access_token=os.environ["MAPILLARY_TOKEN"],
#     output_dir="/content/data/mapillary",
# )
#
# # Santiago centro: ~2000 imágenes para un fine-tune básico
# stats = scraper.scrape_region(
#     region="barrio_republica",  # ← tu barrio bro
#     max_images=2000,
#     resolution="512",           # 512px para Colab (ahorra tiempo)
# )
# print(stats)

# Si no tienes token, el training usa datos sintéticos automáticamente
print("ℹ Datos: el training usará dataset sintético si no hay datos Mapillary")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 6: Configuración del fine-tune                                    │
# └─────────────────────────────────────────────────────────────────────────┘

import sys
sys.path.insert(0, "/content/santiago-world-model")  # ajustar path

from train_lora import LoRAConfig, train

# ── Configuración para A100 40GB ──────────────────────────────────────────

cfg = LoRAConfig(
    # Modelo
    model_id=MODEL_ID,

    # LoRA — máxima calidad visual
    lora_rank=64,        # rank=64: máximo para A100 40GB
    lora_alpha=128,      # alpha = 2 × rank

    # Targets: todos los attention + FF layers del DiT
    target_modules=[
        "to_q", "to_k", "to_v", "to_out.0",
        "to_add_out",
        "ff.net.0.proj", "ff.net.2",
        "proj_in", "proj_out",
        # Añadir si hay VRAM de sobra:
        # "add_k_proj", "add_v_proj",
    ],

    # Training
    learning_rate=1e-4,
    max_steps=3000,      # ~3h en A100
    batch_size=1,
    gradient_accumulation=4,   # effective batch = 4
    dtype="bf16",              # bf16 es más estable en A100

    # Datos
    data_dir="/content/data/mapillary",
    chunk_frames=49,     # T=49 (múltiplo de 4 + 1 para el VAE)
    image_size=480,      # resolución de entrenamiento
    k_references=3,      # reducido para VRAM

    # SWM
    cross_temporal_min_days=30.0,
    vl_sink_enabled=True,
    delta_vl=5,
    geometric_ref=True,
    semantic_ref=True,

    # Output
    output_dir="/content/checkpoints/swm_scl",
    save_every=500,
    log_every=10,
    seed=42,
)

print("Configuración:")
print(f"  LoRA rank: {cfg.lora_rank}, alpha: {cfg.lora_alpha}")
print(f"  Steps: {cfg.max_steps}")
print(f"  Effective batch: {cfg.batch_size * cfg.gradient_accumulation}")
print(f"  Chunk frames: {cfg.chunk_frames}")
print(f"  Image size: {cfg.image_size}×{cfg.image_size}")
print(f"  VL Sink: {cfg.vl_sink_enabled} (Δ={cfg.delta_vl})")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 7: Verificar VRAM antes de entrenar                               │
# └─────────────────────────────────────────────────────────────────────────┘

import torch

def check_vram():
    if not torch.cuda.is_available():
        print("Sin GPU")
        return

    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    used = torch.cuda.memory_allocated() / 1e9
    free = total - used

    print(f"VRAM Total:    {total:.1f} GB")
    print(f"VRAM Usada:    {used:.2f} GB")
    print(f"VRAM Libre:    {free:.2f} GB")

    if total < 39:
        print("⚠ VRAM < 40GB detectada. Considera reducir rank=32 o image_size=320")
        return False
    print("✓ VRAM suficiente para rank=64")
    return True

torch.cuda.empty_cache()
check_vram()


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 8: ENTRENAR                                                       │
# └─────────────────────────────────────────────────────────────────────────┘

# IMPORTANTE: Esta celda puede tardar 3-4 horas en A100
# Colab Pro+ tiene sesiones de hasta 24h

# Activar xformers si está disponible (mejora velocidad ~20%)
try:
    import xformers
    os.environ["XFORMERS_ENABLED"] = "1"
    print("✓ xformers activado")
except ImportError:
    print("ℹ xformers no disponible (no crítico)")

# Limpiar cache VRAM
torch.cuda.empty_cache()
import gc; gc.collect()

print("Iniciando entrenamiento...")
print("Monitorea VRAM con: !nvidia-smi dmon -s m -d 5")
print("─" * 60)

output_dir = train(cfg)

print("─" * 60)
print(f"✓ Entrenamiento completado")
print(f"Checkpoints en: {output_dir}")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 9: Test de generación básica                                      │
# └─────────────────────────────────────────────────────────────────────────┘

from train_lora import load_finetuned_model, generate_santiago_video
from PIL import Image
import IPython.display as display

device = torch.device("cuda")
dtype = torch.bfloat16

# Cargar modelo con LoRA
pipe = load_finetuned_model(
    base_model_id=MODEL_ID,
    lora_ckpt_dir=f"{cfg.output_dir}/final",
    device=device,
    dtype=dtype,
)

# Generar videos con distintos prompts
prompts = [
    "A sunny afternoon on Av. Libertador Bernardo O'Higgins in Santiago, Chile. "
    "Heavy traffic, pedestrians on sidewalks, tall buildings lining the avenue.",

    "Barrio República in Santiago, Chile. "
    "Narrow tree-lined street, colonial architecture, warm golden hour light.",

    "A massive earthquake cracks the streets of Santiago, Chile. "
    "Dramatic dust clouds, people running, broken asphalt.",

    "Santiago, Chile under heavy rain. "
    "Wet streets reflecting city lights, overcast sky, fog on the Andes.",
]

for i, prompt in enumerate(prompts):
    print(f"\nGenerando prompt {i+1}/4:")
    print(f"  '{prompt[:80]}...'")

    frames = generate_santiago_video(
        pipe=pipe,
        prompt=prompt,
        num_frames=49,
        guidance_scale=7.0,
        num_inference_steps=50,
        seed=42 + i,
    )

    # Guardar primer frame
    output_path = f"/content/output_{i+1}.jpg"
    frames[0].save(output_path)
    print(f"  Guardado: {output_path}")

    # Mostrar en notebook
    display.display(display.Image(output_path))

print("\n✓ Generación completada")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 10: Exportar a Google Drive (para no perder el checkpoint)        │
# └─────────────────────────────────────────────────────────────────────────┘

# from google.colab import drive
# drive.mount('/content/drive')

# import shutil
# shutil.copytree(
#     cfg.output_dir,
#     f"/content/drive/MyDrive/swm_scl_checkpoints",
# )
# print("✓ Checkpoints guardados en Drive")

# También puedes descargar los pesos LoRA directamente:
# Son solo ~200-400MB (mucho más pequeños que el modelo completo)
# from google.colab import files
# files.download(f"{cfg.output_dir}/final/lora_weights/adapter_model.safetensors")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 11: Estimación de VRAM por configuración                          │
# └─────────────────────────────────────────────────────────────────────────┘

VRAM_ESTIMATES = """
╔════════════════════════════════════════════════════════════════╗
║         VRAM ESTIMADA SEGÚN CONFIGURACIÓN                      ║
╠════════════════╦══════════╦═══════════╦════════════════════════╣
║ GPU            ║ LoRA rank ║ Image size ║ VRAM estimada         ║
╠════════════════╬══════════╬═══════════╬════════════════════════╣
║ A100 40GB      ║ 64        ║ 480px     ║ ~32-38GB ✓ (este)     ║
║ A100 40GB      ║ 32        ║ 480px     ║ ~25-30GB ✓ holgado    ║
║ A100 40GB      ║ 64        ║ 320px     ║ ~22-26GB ✓ rápido     ║
║ RTX 4090 24GB  ║ 32        ║ 320px     ║ ~20-23GB ✓            ║
║ RTX 3090 24GB  ║ 16        ║ 256px     ║ ~18-22GB ✓ lento      ║
║ RTX 3080 10GB  ║ 8         ║ 256px     ║ ~8-10GB ⚠ muy justo   ║
║ CPU only       ║ cualquier ║ cualquier ║ posible (lentísimo)   ║
╚════════════════╩══════════╩═══════════╩════════════════════════╝

Consejos para ahorrar VRAM:
  1. Reducir lora_rank: 64 → 32 → 16
  2. Reducir image_size: 480 → 320 → 256
  3. Habilitar gradient_checkpointing=True (ya activado)
  4. VAE slicing + tiling (ya activado)
  5. Reducir chunk_frames: 49 → 25 → 13
  6. batch_size=1 siempre (ya configurado)
"""
print(VRAM_ESTIMATES)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CELDA 12: Quick test sin datos reales (solo para verificar que corre)   │
# └─────────────────────────────────────────────────────────────────────────┘

def quick_test(cfg: LoRAConfig):
    """
    Test rápido del pipeline de entrenamiento con datos sintéticos.
    Solo 10 pasos para verificar que todo funciona antes del run largo.
    """
    print("Quick test: 10 pasos con datos sintéticos...")

    test_cfg = LoRAConfig(
        model_id=cfg.model_id,
        lora_rank=8,           # rank bajo para el test
        lora_alpha=16,
        max_steps=10,
        batch_size=1,
        gradient_accumulation=1,
        dtype=cfg.dtype,
        image_size=256,        # resolución baja para el test
        chunk_frames=13,       # pocos frames
        output_dir="/content/test_ckpt",
        save_every=999999,     # no guardar durante test
        log_every=1,
    )

    try:
        train(test_cfg)
        print("✓ Quick test exitoso — el pipeline funciona correctamente")
        return True
    except Exception as e:
        print(f"✗ Quick test falló: {e}")
        import traceback; traceback.print_exc()
        return False

# Descomentar para correr el test:
# quick_test(cfg)
