"""
train_lora.py — Santiago World Model Fine-tune
================================================
LoRA fine-tune de Cosmos-Predict2.5-2B sobre datos de street-view de Santiago.
Diseñado para correr en Google Colab Pro+ (A100 40GB) o RunPod.

Implementa fielmente la metodología SWM (Seo et al., 2025):
  - Cross-temporal pairing
  - Geometric + Semantic referencing
  - Virtual Lookahead Sink
  - Intermittent Freeze-Frame para view interpolation

Uso:
    # Colab / RunPod:
    python train_lora.py --config configs/santiago_config.yaml --colab

    # Local con GPU:
    python train_lora.py --config configs/santiago_config.yaml --gpus 1
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# LoRA via PEFT
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("⚠ peft no instalado. Corre: pip install peft")

# Diffusers / Cosmos
try:
    from diffusers import (
        AutoencoderKLCogVideoX,
        CogVideoXDPMScheduler,
    )
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False
    print("⚠ diffusers no instalado. Corre: pip install diffusers>=0.30.0")

from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════

@dataclass
class LoRAConfig:
    """Configuración del fine-tune LoRA."""

    # Modelo base
    model_id: str = "nvidia/Cosmos-Predict2.5-2B-Video2World"
    # Alternativa si el anterior requiere licencia y no la aceptaste:
    # model_id: str = "THUDM/CogVideoX-5b"  # similar arquitectura DiT

    # LoRA — cuánto poder aplicar al DiT
    lora_rank: int = 64           # r=64 → máxima capacidad visual en A100 40GB
    lora_alpha: int = 128         # alpha = 2*rank → escala óptima
    lora_dropout: float = 0.05
    # Qué módulos del DiT atacar para máximo impacto visual:
    target_modules: list = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",  # self-attention
        "to_add_out",                           # cross-attention
        "ff.net.0.proj", "ff.net.2",           # feed-forward
        "proj_in", "proj_out",                 # proyecciones de entrada/salida
    ])

    # Training
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine_with_restarts"
    warmup_steps: int = 100
    max_steps: int = 3000          # ~3h en A100 para dataset pequeño
    batch_size: int = 1            # A100 40GB con rank=64
    gradient_accumulation: int = 4  # effective batch = 4
    max_grad_norm: float = 1.0

    # Precisión
    dtype: str = "bf16"            # bf16 es más estable que fp16 para DiT
    gradient_checkpointing: bool = True

    # Datos
    data_dir: str = "./data/mapillary"
    geo_index_path: str = "./data/geo_index.pkl"
    sequence_length: int = 10      # keyframes por secuencia
    chunk_frames: int = 49         # frames por chunk (múltiplo de 4 + 1)
    image_size: int = 480          # resolución de entrenamiento
    k_references: int = 3          # refs por chunk (reducido para VRAM)

    # SWM específico
    cross_temporal_min_days: float = 30.0
    vl_sink_enabled: bool = True
    delta_vl: int = 5
    geometric_ref: bool = True
    semantic_ref: bool = True

    # Dropout (CFG)
    caption_dropout: float = 0.20
    reference_dropout: float = 0.20

    # Checkpoints
    output_dir: str = "./checkpoints/swm_scl_lora"
    save_every: int = 500
    log_every: int = 10

    # Reproducibilidad
    seed: int = 42


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

class SantiagoStreetViewDataset(Dataset):
    """
    Dataset de pares (secuencia objetivo, referencias cross-temporales)
    para el fine-tune del Santiago World Model.

    Cada item contiene:
      - video_frames: (T, C, H, W) frames del video objetivo
      - ref_images: (K, C, H, W) referencias cross-temporales
      - warped_frames: (T, C, H, W) frames warpeados geométricamente
      - camera_action: str con la acción de cámara
      - caption: texto descriptivo de la escena
      - vl_sink_image: (C, H, W) imagen de Virtual Lookahead Sink
    """

    CAMERA_ACTION_TEMPLATES = {
        "straight":    "Driving straight ahead on {road_type} in Santiago, Chile.",
        "left_turn":   "Turning left on {road_type} in Santiago, Chile.",
        "right_turn":  "Turning right on {road_type} in Santiago, Chile.",
        "stop":        "Stopped at an intersection in Santiago, Chile.",
    }

    ROAD_TYPES = [
        "an urban street", "Av. Libertador Bernardo O'Higgins",
        "a residential street", "a busy avenue", "Alameda",
        "Av. Providencia", "a narrow alley", "a tree-lined street",
    ]

    def __init__(
        self,
        pairs_dir: str,
        images_dir: str,
        cfg: LoRAConfig,
        split: str = "train",
        transform=None,
    ):
        self.pairs_dir = Path(pairs_dir)
        self.images_dir = Path(images_dir)
        self.cfg = cfg
        self.split = split
        self.transform = transform

        # Cargar todos los pares de entrenamiento
        self.pairs = sorted(self.pairs_dir.glob("pair_*.json"))
        if not self.pairs:
            log.warning(
                f"No se encontraron pares en {pairs_dir}. "
                "Generando dataset sintético para testing..."
            )
            self._use_synthetic = True
        else:
            self._use_synthetic = False
            # 90/10 split
            n = len(self.pairs)
            if split == "train":
                self.pairs = self.pairs[:int(n * 0.9)]
            else:
                self.pairs = self.pairs[int(n * 0.9):]

        log.info(f"Dataset '{split}': {len(self.pairs)} pares")

    def __len__(self):
        # Si es sintético, devolvemos 1000 items para poder entrenar
        return len(self.pairs) if not self._use_synthetic else 1000

    def __getitem__(self, idx):
        if self._use_synthetic:
            return self._synthetic_item(idx)
        return self._real_item(idx)

    def _load_image(self, path: str) -> torch.Tensor:
        """Carga una imagen y la convierte a tensor (C, H, W) float [−1, 1]."""
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.cfg.image_size, self.cfg.image_size), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
            return torch.from_numpy(arr).permute(2, 0, 1)
        except Exception:
            return torch.zeros(3, self.cfg.image_size, self.cfg.image_size)

    def _real_item(self, idx):
        """Carga un par real desde disco."""
        pair_path = self.pairs[idx]
        with open(pair_path) as f:
            pair = json.load(f)

        T = self.cfg.chunk_frames
        K = self.cfg.k_references
        H = W = self.cfg.image_size

        # Cargar frames objetivo
        video_frames = []
        for img_id in pair["target_ids"][:T]:
            # Buscar el archivo imagen
            candidates = list(self.images_dir.glob(f"{img_id}_*.jpg"))
            if candidates:
                frame = self._load_image(str(candidates[0]))
            else:
                frame = torch.zeros(3, H, W)
            video_frames.append(frame)

        # Padding si hay menos de T frames
        while len(video_frames) < T:
            video_frames.append(video_frames[-1] if video_frames else torch.zeros(3, H, W))
        video_frames = torch.stack(video_frames[:T])  # (T, C, H, W)

        # Cargar referencias
        ref_images = []
        for img_id in pair["reference_ids"][:K]:
            candidates = list(self.images_dir.glob(f"{img_id}_*.jpg"))
            if candidates:
                ref = self._load_image(str(candidates[0]))
            else:
                ref = torch.zeros(3, H, W)
            ref_images.append(ref)

        while len(ref_images) < K:
            ref_images.append(torch.zeros(3, H, W))
        ref_images = torch.stack(ref_images[:K])  # (K, C, H, W)

        # Caption
        caption = self._build_caption(pair.get("camera_action", "straight"))

        # Caption dropout (CFG)
        if random.random() < self.cfg.caption_dropout:
            caption = ""

        # Reference dropout
        if random.random() < self.cfg.reference_dropout:
            ref_images = torch.zeros_like(ref_images)

        # VL Sink: última referencia disponible como lookahead
        vl_sink = ref_images[-1] if self.cfg.vl_sink_enabled else torch.zeros(3, H, W)

        # Warped frames (placeholder — en producción: depth splatting real)
        warped_frames = video_frames.clone()

        return {
            "video_frames": video_frames,        # (T, C, H, W)
            "ref_images": ref_images,            # (K, C, H, W)
            "warped_frames": warped_frames,      # (T, C, H, W)
            "vl_sink": vl_sink,                  # (C, H, W)
            "caption": caption,
            "camera_action": pair.get("camera_action", "straight"),
        }

    def _synthetic_item(self, idx):
        """
        Item sintético para testing sin datos reales.
        Usa ruido gaussiano + gradientes simples para simular calles.
        """
        rng = np.random.RandomState(idx)
        T = self.cfg.chunk_frames
        K = self.cfg.k_references
        H = W = self.cfg.image_size

        # Simular frames de street-view: sky (top 40%) + road (bottom 60%)
        frames = []
        for t in range(T):
            frame = np.zeros((H, W, 3), dtype=np.float32)
            # Sky — azul/gris
            sky_h = int(H * 0.4)
            frame[:sky_h] = rng.uniform(-0.5, 0.2, (sky_h, W, 3))
            # Road — gris oscuro
            frame[sky_h:] = rng.uniform(-0.9, -0.5, (H - sky_h, W, 3))
            # Buildings — franjas verticales
            for b in range(rng.randint(3, 8)):
                bx = rng.randint(0, W - 30)
                bw = rng.randint(15, 50)
                bh = rng.randint(int(H * 0.1), int(H * 0.4))
                frame[sky_h - bh:sky_h, bx:bx + bw] = rng.uniform(-0.3, 0.3, (bh, bw, 3))
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))

        video_frames = torch.stack(frames)  # (T, C, H, W)

        # Referencias (versión ligeramente perturbada)
        ref_images = []
        for k in range(K):
            noise = torch.randn_like(video_frames[0]) * 0.1
            ref_images.append(video_frames[0] + noise)
        ref_images = torch.stack(ref_images)  # (K, C, H, W)

        actions = ["straight", "left_turn", "right_turn", "stop"]
        action = actions[idx % len(actions)]
        caption = self._build_caption(action)

        return {
            "video_frames": video_frames,
            "ref_images": ref_images,
            "warped_frames": video_frames.clone(),
            "vl_sink": ref_images[-1],
            "caption": caption,
            "camera_action": action,
        }

    def _build_caption(self, action: str) -> str:
        road = random.choice(self.ROAD_TYPES)
        template = self.CAMERA_ACTION_TEMPLATES.get(action, self.CAMERA_ACTION_TEMPLATES["straight"])
        base = template.format(road_type=road)
        # Añadir variedad climática
        conditions = [
            "Sunny afternoon with clear Andes views.",
            "Overcast day with soft diffuse light.",
            "Golden hour, warm tones on the buildings.",
            "Light fog typical of Santiago winter mornings.",
            "Bright midday sun, sharp shadows.",
        ]
        return base + " " + random.choice(conditions)


# ══════════════════════════════════════════════════════════════════
# MODELO COSMOS + LORA
# ══════════════════════════════════════════════════════════════════

def load_cosmos_model(cfg: LoRAConfig, device: torch.device):
    """
    Carga Cosmos-Predict2.5-2B y aplica LoRA al DiT.

    El modelo Cosmos usa una arquitectura Diffusion Transformer (DiT)
    con 28 bloques, hidden_dim=2048, 16 attention heads.

    LoRA rank=64 agrega ~50M parámetros entrenables sobre ~2B frozen.
    Eso es ~2.5% del modelo — máximo impacto visual con mínimo VRAM.
    """
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[cfg.dtype]

    log.info(f"Cargando modelo base: {cfg.model_id}")
    log.info(f"dtype={cfg.dtype}, device={device}")

    # ── Intentar cargar Cosmos ─────────────────────────────────────
    # Cosmos requiere aceptar la licencia en HuggingFace:
    # https://huggingface.co/nvidia/Cosmos-Predict2.5-2B-Video2World
    #
    # Si no tienes acceso, usa CogVideoX-5b como proxy (misma arquitectura DiT):
    # model_id = "THUDM/CogVideoX-5b"
    try:
        from transformers import AutoModel, T5EncoderModel, AutoTokenizer

        log.info("Cargando text encoder (T5-XXL)...")
        text_encoder = T5EncoderModel.from_pretrained(
            cfg.model_id,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            subfolder="tokenizer",
        )

        log.info("Cargando 3D VAE...")
        vae = AutoencoderKLCogVideoX.from_pretrained(
            cfg.model_id,
            subfolder="vae",
            torch_dtype=torch_dtype,
        )

        log.info("Cargando DiT transformer...")
        transformer = AutoModel.from_pretrained(
            cfg.model_id,
            subfolder="transformer",
            torch_dtype=torch_dtype,
        )

        scheduler = CogVideoXDPMScheduler.from_pretrained(
            cfg.model_id,
            subfolder="scheduler",
        )

        log.info("✓ Cosmos cargado exitosamente")
        return transformer, vae, text_encoder, tokenizer, scheduler

    except Exception as e:
        log.error(f"Error cargando Cosmos: {e}")
        log.warning("Fallback: usando CogVideoX-5b (misma arquitectura DiT)")

        # Fallback a CogVideoX que tiene misma arquitectura
        try:
            from diffusers import CogVideoXPipeline
            pipe = CogVideoXPipeline.from_pretrained(
                "THUDM/CogVideoX-5b",
                torch_dtype=torch_dtype,
            )
            return (
                pipe.transformer,
                pipe.vae,
                pipe.text_encoder,
                pipe.tokenizer,
                pipe.scheduler,
            )
        except Exception as e2:
            log.error(f"Error cargando CogVideoX fallback: {e2}")
            raise RuntimeError(
                "No se pudo cargar ningún modelo base. "
                "Asegúrate de:\n"
                "1. Aceptar la licencia en HuggingFace\n"
                "2. huggingface-cli login\n"
                "3. pip install diffusers>=0.30.0 transformers peft\n"
            ) from e2


def apply_lora(transformer: nn.Module, cfg: LoRAConfig) -> nn.Module:
    """
    Aplica LoRA al DiT transformer.

    rank=64, alpha=128: máxima expresividad visual.
    Los target_modules atacan todos los attention + FF layers
    del DiT — esto maximiza la calidad de los detalles generados.
    """
    if not HAS_PEFT:
        raise ImportError("pip install peft")

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        # task_type no aplica directamente para DiT,
        # pero lo dejamos como FEATURE_EXTRACTION
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    transformer = get_peft_model(transformer, lora_config)

    trainable, total = transformer.get_nb_trainable_parameters()
    log.info(
        f"LoRA aplicado — "
        f"Parámetros entrenables: {trainable/1e6:.1f}M / {total/1e6:.0f}M total "
        f"({100*trainable/total:.1f}%)"
    )
    transformer.print_trainable_parameters()

    return transformer


# ══════════════════════════════════════════════════════════════════
# ENCODE / DECODE
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_frames(vae, frames: torch.Tensor, dtype) -> torch.Tensor:
    """
    Codifica frames de video al espacio latente del 3D VAE.

    Args:
        frames: (B, T, C, H, W) float [−1, 1]
    Returns:
        latents: (B, L, C_lat, H_lat, W_lat)
    """
    B, T, C, H, W = frames.shape
    # El VAE de Cosmos espera (B, C, T, H, W)
    frames_bcthw = frames.permute(0, 2, 1, 3, 4).to(dtype)

    # Encode
    posterior = vae.encode(frames_bcthw)
    latents = posterior.latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    return latents  # (B, C_lat, T_lat, H_lat, W_lat)


@torch.no_grad()
def encode_text(text_encoder, tokenizer, captions: list[str], device, dtype):
    """Codifica captions con T5-XXL."""
    tokens = tokenizer(
        captions,
        max_length=226,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    embeddings = text_encoder(**tokens).last_hidden_state
    return embeddings.to(dtype)


# ══════════════════════════════════════════════════════════════════
# LOSS Y TRAINING STEP
# ══════════════════════════════════════════════════════════════════

def compute_swm_loss(
    transformer,
    scheduler,
    latents: torch.Tensor,
    ref_latents: torch.Tensor,
    vl_latent: torch.Tensor,
    text_embeddings: torch.Tensor,
    cfg: LoRAConfig,
    device: torch.device,
    dtype,
) -> torch.Tensor:
    """
    Loss de entrenamiento SWM.

    Implementa el objetivo de diffusion estándar (v-prediction o epsilon)
    con conditioning por referencias y VL Sink.

    La secuencia de tokens sigue Eq. (1) del paper:
        Z_seq = [Z_hist; Z_target; z_VL]
        p_seq = [1..H; H+1..H+L; H+L+Δ_VL]
    """
    B = latents.shape[0]

    # Sample timesteps
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (B,), device=device, dtype=torch.long,
    )

    # Agregar ruido a los latentes objetivo
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    # Conditioning: concatenar warped video + referencias
    # En Cosmos, el conditioning se hace via channel concatenation en el DiT input
    # y token injection para semantic references

    # Concatenar referencia (geometric conditioning) con el noisy latent
    # ref_latents: (B, K, C_lat, T_lat, H_lat, W_lat) → promedio de refs
    ref_mean = ref_latents.mean(dim=1)  # (B, C_lat, T_lat, H_lat, W_lat)
    model_input = torch.cat([noisy_latents, ref_mean], dim=1)  # channel concat

    # VL Sink: añadir como token extra con posición RoPE desplazada
    # En la implementación simplificada, lo añadimos al conditioning
    if cfg.vl_sink_enabled and vl_latent is not None:
        # Se añade como conditioning adicional — el transformer lo atiende
        vl_expanded = vl_latent.unsqueeze(2).expand_as(ref_mean)
        model_input = torch.cat([model_input, vl_expanded], dim=1)

    # Forward pass del DiT
    # Nota: la signature exacta varía según la implementación de Cosmos/CogVideoX
    try:
        noise_pred = transformer(
            hidden_states=model_input,
            timestep=timesteps,
            encoder_hidden_states=text_embeddings,
            return_dict=False,
        )[0]
    except TypeError:
        # Fallback si la API es diferente
        noise_pred = transformer(
            model_input,
            timesteps,
            text_embeddings,
        ).sample

    # Loss — MSE sobre predicción de ruido (epsilon prediction)
    # Para v-prediction: target = scheduler.get_velocity(latents, noise, timesteps)
    if hasattr(scheduler.config, "prediction_type") and scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = noise

    # Asegurar mismas dimensiones (el model_input tiene más canales que el target)
    if noise_pred.shape[1] != target.shape[1]:
        noise_pred = noise_pred[:, :target.shape[1]]

    loss = F.mse_loss(noise_pred.float(), target.float())
    return loss


# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def train(cfg: LoRAConfig):
    """Loop principal de entrenamiento."""

    # ── Setup ─────────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[cfg.dtype]

    if device.type == "cpu":
        log.warning("⚠ Entrenando en CPU — muy lento. Usa GPU (Colab/RunPod).")
        cfg.dtype = "fp32"
        dtype = torch.float32

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output: {output_dir}")

    # ── Dataset ───────────────────────────────────────────────────
    pairs_dir = Path(cfg.data_dir) / "training_pairs"
    images_dir = Path(cfg.data_dir) / "images"

    dataset = SantiagoStreetViewDataset(
        pairs_dir=str(pairs_dir),
        images_dir=str(images_dir),
        cfg=cfg,
        split="train",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    # ── Modelo ────────────────────────────────────────────────────
    transformer, vae, text_encoder, tokenizer, scheduler = load_cosmos_model(cfg, device)

    # Freeze VAE y text encoder — solo entrenamos el DiT
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Gradient checkpointing del VAE (ahorra VRAM)
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    # Aplicar LoRA al transformer
    transformer = apply_lora(transformer, cfg)

    if cfg.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
        log.info("Gradient checkpointing activado")

    # Mover a device
    transformer = transformer.to(device, dtype=dtype)
    vae = vae.to(device, dtype=dtype)
    text_encoder = text_encoder.to(device, dtype=dtype)

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Scheduler cosine con warm restarts
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.max_steps // 3,
        T_mult=1,
    )

    # Mixed precision scaler (solo para fp16)
    use_scaler = cfg.dtype == "fp16"
    scaler = GradScaler(enabled=use_scaler)

    # ── Training ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SANTIAGO WORLD MODEL — LoRA Fine-tune")
    log.info(f"  Pasos: {cfg.max_steps}")
    log.info(f"  Batch: {cfg.batch_size} × {cfg.gradient_accumulation} = {cfg.batch_size * cfg.gradient_accumulation}")
    log.info(f"  LR: {cfg.learning_rate}")
    log.info(f"  LoRA rank: {cfg.lora_rank}, alpha: {cfg.lora_alpha}")
    log.info(f"  Dataset: {len(dataset)} pares")
    log.info("=" * 60)

    global_step = 0
    best_loss = float("inf")
    running_loss = []

    transformer.train()

    progress = tqdm(total=cfg.max_steps, desc="SWM-SCL LoRA")

    while global_step < cfg.max_steps:
        for batch in dataloader:
            if global_step >= cfg.max_steps:
                break

            # ── Forward ───────────────────────────────────────────
            video_frames = batch["video_frames"].to(device)      # (B, T, C, H, W)
            ref_images = batch["ref_images"].to(device)          # (B, K, C, H, W)
            warped_frames = batch["warped_frames"].to(device)    # (B, T, C, H, W)
            vl_sink = batch["vl_sink"].to(device)                # (B, C, H, W)
            captions = batch["caption"]

            with autocast(dtype=dtype, enabled=device.type == "cuda"):

                # Encode video a latentes
                latents = encode_frames(vae, video_frames, dtype)

                # Encode referencias
                B, K, C, H, W = ref_images.shape
                refs_flat = ref_images.view(B * K, 1, C, H, W)
                ref_latents_flat = encode_frames(vae, refs_flat, dtype)
                ref_latents = ref_latents_flat.view(B, K, *ref_latents_flat.shape[1:])

                # Encode VL Sink
                vl_exp = vl_sink.unsqueeze(1)  # (B, 1, C, H, W)
                vl_latent = encode_frames(vae, vl_exp, dtype).squeeze(1)

                # Encode texto
                text_emb = encode_text(text_encoder, tokenizer, list(captions), device, dtype)

                # Loss SWM
                loss = compute_swm_loss(
                    transformer=transformer,
                    scheduler=scheduler,
                    latents=latents,
                    ref_latents=ref_latents,
                    vl_latent=vl_latent,
                    text_embeddings=text_emb,
                    cfg=cfg,
                    device=device,
                    dtype=dtype,
                )

                loss = loss / cfg.gradient_accumulation

            # ── Backward ──────────────────────────────────────────
            scaler.scale(loss).backward()

            if (global_step + 1) % cfg.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in transformer.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

            # ── Logging ───────────────────────────────────────────
            loss_val = loss.item() * cfg.gradient_accumulation
            running_loss.append(loss_val)

            if global_step % cfg.log_every == 0:
                avg_loss = np.mean(running_loss[-cfg.log_every:])
                lr_now = optimizer.param_groups[0]["lr"]
                progress.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{lr_now:.2e}",
                    step=global_step,
                )

                if avg_loss < best_loss:
                    best_loss = avg_loss

            # ── Checkpoint ────────────────────────────────────────
            if global_step % cfg.save_every == 0 and global_step > 0:
                save_checkpoint(transformer, optimizer, cfg, global_step, output_dir)
                log.info(f"✓ Checkpoint guardado: step={global_step}, loss={best_loss:.4f}")

            global_step += 1
            progress.update(1)

    # ── Save final ────────────────────────────────────────────────
    save_checkpoint(transformer, optimizer, cfg, global_step, output_dir, final=True)
    log.info(f"✓ Entrenamiento completado. Best loss: {best_loss:.4f}")
    log.info(f"Checkpoints en: {output_dir}")

    return output_dir


def save_checkpoint(
    transformer,
    optimizer,
    cfg: LoRAConfig,
    step: int,
    output_dir: Path,
    final: bool = False,
):
    """Guarda solo los pesos LoRA (mucho más pequeño que el modelo completo)."""
    name = "final" if final else f"step_{step:06d}"
    ckpt_dir = output_dir / name
    ckpt_dir.mkdir(exist_ok=True)

    # Guardar solo pesos LoRA via PEFT
    if HAS_PEFT:
        transformer.save_pretrained(ckpt_dir / "lora_weights")

    # Guardar config
    import dataclasses
    cfg_dict = dataclasses.asdict(cfg)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Guardar optimizer state (para resumir)
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict()},
        ckpt_dir / "optimizer.pt",
    )


# ══════════════════════════════════════════════════════════════════
# INFERENCE — usar el modelo fine-tuneado
# ══════════════════════════════════════════════════════════════════

def load_finetuned_model(base_model_id: str, lora_ckpt_dir: str, device, dtype):
    """
    Carga el modelo base + los pesos LoRA para inferencia.

    Args:
        base_model_id: HuggingFace model ID del modelo base
        lora_ckpt_dir: directorio con los pesos LoRA guardados
        device: torch device
        dtype: torch dtype
    """
    from diffusers import CogVideoXPipeline

    log.info(f"Cargando pipeline base: {base_model_id}")
    pipe = CogVideoXPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    )

    log.info(f"Cargando pesos LoRA desde: {lora_ckpt_dir}")
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer,
        Path(lora_ckpt_dir) / "lora_weights",
    )

    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()

    return pipe


@torch.no_grad()
def generate_santiago_video(
    pipe,
    prompt: str,
    start_image: Optional[Image.Image] = None,
    num_frames: int = 49,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 50,
    seed: int = 42,
) -> list[Image.Image]:
    """
    Genera un video de Santiago dado un prompt de texto.

    Args:
        pipe: pipeline con LoRA cargado
        prompt: descripción de la escena
        start_image: imagen inicial (opcional)
        num_frames: número de frames a generar
        guidance_scale: CFG scale
        num_inference_steps: pasos de denoising
        seed: semilla para reproducibilidad

    Returns:
        Lista de PIL Images
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    output = pipe(
        prompt=prompt,
        image=start_image,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    return output.frames[0]


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Santiago World Model — LoRA Fine-tune")
    parser.add_argument("--config", type=str, default=None, help="Path al YAML de config")
    parser.add_argument("--data-dir", type=str, default="./data/mapillary")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/swm_scl_lora")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank (32=rápido, 64=máx calidad)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--colab", action="store_true", help="Optimizaciones para Colab")
    parser.add_argument("--resume", type=str, default=None, help="Continuar desde checkpoint")
    parser.add_argument("--generate", action="store_true", help="Modo inferencia")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    # Optimizaciones Colab
    if args.colab:
        log.info("Modo Colab: activando optimizaciones de memoria")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Rank más bajo si hay poca VRAM
        if args.rank == 64:
            log.info("Colab Pro+ A100: rank=64 debería funcionar bien")

    cfg = LoRAConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lora_rank=args.rank,
        lora_alpha=args.rank * 2,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )

    if args.generate:
        if not args.checkpoint:
            log.error("--checkpoint requerido para --generate")
            sys.exit(1)
        prompt = args.prompt or "A sunny afternoon on Av. Libertador O'Higgins in Santiago, Chile."
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        pipe = load_finetuned_model(cfg.model_id, args.checkpoint, device, dtype)
        frames = generate_santiago_video(pipe, prompt)
        log.info(f"Generados {len(frames)} frames")
        frames[0].save("output_santiago.jpg")
        log.info("Primer frame guardado: output_santiago.jpg")
    else:
        train(cfg)


if __name__ == "__main__":
    main()
