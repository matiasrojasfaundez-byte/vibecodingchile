"""
data_pipeline/cross_temporal_pairer.py
========================================
Implementa cross-temporal pairing (Sec. 3.1 del paper SWM).

La idea clave: los pares (referencia, video objetivo) deben venir de
TIMESTAMPS DISTINTOS para que el modelo aprenda a ignorar objetos
dinámicos (autos, peatones) y enfocarse en estructura persistente.

También contiene el pipeline de view interpolation (Sec. 3.1):
- Estrategia Intermittent Freeze-Frame para síntesis de video
  continuo a partir de keyframes de street-view dispersos.
"""

import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone

import numpy as np

from .geo_indexer import StreetViewRef, GeoIndexer

log = logging.getLogger(__name__)


# ─── Estructuras de datos ────────────────────────────────────────────────────

@dataclass
class TrainingPair:
    """
    Par de entrenamiento (referencias, video objetivo).
    
    target_sequence: N keyframes consecutivos del video objetivo
    reference_images: K imágenes de street-view como referencia
    timestamp_gap_days: diferencia temporal entre refs y target
    """
    target_sequence: list[StreetViewRef]
    reference_images: list[StreetViewRef]
    timestamp_gap_days: float
    camera_action: str  # "straight" | "left_turn" | "right_turn" | "stop"
    waypoints: list[tuple[float, float]]  # (lat, lng) de cada keyframe


# ─── Cross-Temporal Pairer ────────────────────────────────────────────────────

class CrossTemporalPairer:
    """
    Genera pares de entrenamiento con cross-temporal pairing.
    
    Para cada secuencia objetivo, busca referencias espacialmente
    cercanas pero temporalmente distintas (diferente fecha de captura).
    
    Esto fuerza al modelo a:
    1. Aprender estructura persistente (edificios, calles)
    2. Ignorar contenido transitorio (vehículos, peatones)
    """

    def __init__(
        self,
        indexer: GeoIndexer,
        min_timestamp_gap_days: float = 30.0,   # 1 mes mínimo
        max_timestamp_gap_days: float = 730.0,  # 2 años máximo
        k_references: int = 5,
        retrieval_radius_m: float = 150.0,
        sequence_length: int = 10,
    ):
        self.indexer = indexer
        self.min_gap = min_timestamp_gap_days * 86400 * 1000  # a ms
        self.max_gap = max_timestamp_gap_days * 86400 * 1000
        self.k = k_references
        self.radius = retrieval_radius_m
        self.seq_len = sequence_length

    def compute_camera_action(self, waypoints: list[tuple[float, float]]) -> str:
        """
        Determina la acción de cámara dominante de una secuencia.
        Basado en el heading promedio entre waypoints consecutivos.
        
        Returns: "straight" | "left_turn" | "right_turn" | "stop"
        """
        if len(waypoints) < 2:
            return "stop"

        headings = []
        for i in range(1, len(waypoints)):
            lat1, lng1 = waypoints[i-1]
            lat2, lng2 = waypoints[i]
            dlng = np.radians(lng2 - lng1)
            x = np.sin(dlng) * np.cos(np.radians(lat2))
            y = (np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) -
                 np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(dlng))
            heading = (np.degrees(np.arctan2(x, y)) + 360) % 360
            headings.append(heading)

        # Cambio de heading entre inicio y fin
        if len(headings) < 2:
            return "straight"

        delta = headings[-1] - headings[0]
        # Normalizar a [-180, 180]
        delta = ((delta + 180) % 360) - 180

        if abs(delta) < 15:
            return "straight"
        elif delta > 15:
            return "right_turn"
        elif delta < -15:
            return "left_turn"
        else:
            return "straight"

    def is_valid_temporal_gap(
        self,
        target_ts: int,
        ref_ts: int,
    ) -> bool:
        """Verifica que la brecha temporal está en el rango permitido."""
        gap = abs(target_ts - ref_ts)
        return self.min_gap <= gap <= self.max_gap

    def pair_sequence(
        self,
        target_sequence: list[StreetViewRef],
        rng: Optional[random.Random] = None,
    ) -> Optional[TrainingPair]:
        """
        Genera un par de entrenamiento para una secuencia objetivo.
        
        Args:
            target_sequence: secuencia de refs que forman el video objetivo
            rng: generador aleatorio para reproducibilidad
        
        Returns:
            TrainingPair o None si no se encontraron referencias válidas
        """
        if rng is None:
            rng = random.Random()

        # Usar el timestamp promedio de la secuencia como "tiempo objetivo"
        target_ts = int(np.mean([r.captured_at for r in target_sequence if r.captured_at]))

        waypoints = [(r.lat, r.lng) for r in target_sequence]

        # Buscar referencias para cada keyframe de la secuencia
        # Primero buscar candidates sin filtro temporal, luego filtrar
        all_candidates: list[StreetViewRef] = []

        for lat, lng in waypoints:
            candidates = self.indexer.query(
                lat=lat,
                lng=lng,
                k=self.k * 3,  # Pedir más para tener donde filtrar
                max_dist_m=self.radius,
            )
            for ref, _ in candidates:
                # Cross-temporal: verificar gap de tiempo
                if ref.captured_at and self.is_valid_temporal_gap(target_ts, ref.captured_at):
                    all_candidates.append(ref)

        if not all_candidates:
            log.debug(f"Sin referencias cross-temporales para secuencia en {waypoints[0]}")
            return None

        # Deduplicar y seleccionar K referencias
        seen_ids = set()
        unique_refs = []
        for ref in all_candidates:
            if ref.image_id not in seen_ids:
                seen_ids.add(ref.image_id)
                unique_refs.append(ref)

        # Shuffle para variedad + tomar K
        rng.shuffle(unique_refs)
        selected_refs = unique_refs[:self.k]

        # Calcular gap temporal promedio en días
        gaps = []
        for ref in selected_refs:
            if ref.captured_at:
                gap_days = abs(target_ts - ref.captured_at) / (86400 * 1000)
                gaps.append(gap_days)
        avg_gap = np.mean(gaps) if gaps else 0.0

        camera_action = self.compute_camera_action(waypoints)

        return TrainingPair(
            target_sequence=target_sequence,
            reference_images=selected_refs,
            timestamp_gap_days=avg_gap,
            camera_action=camera_action,
            waypoints=waypoints,
        )

    def generate_pairs_from_sequences(
        self,
        sequences: list[list[StreetViewRef]],
        output_dir: str = "./data/training_pairs",
    ) -> dict:
        """
        Genera y guarda pares de entrenamiento para un conjunto de secuencias.
        
        Args:
            sequences: lista de secuencias de StreetViewRef
            output_dir: directorio de salida
        
        Returns:
            Estadísticas del proceso
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pairs = []
        failed = 0
        action_counts = {"straight": 0, "left_turn": 0, "right_turn": 0, "stop": 0}

        for i, seq in enumerate(sequences):
            pair = self.pair_sequence(seq)
            if pair is None:
                failed += 1
                continue

            pairs.append(pair)
            action_counts[pair.camera_action] += 1

            # Guardar metadata del par
            pair_meta = {
                "id": i,
                "target_ids": [r.image_id for r in pair.target_sequence],
                "reference_ids": [r.image_id for r in pair.reference_images],
                "timestamp_gap_days": pair.timestamp_gap_days,
                "camera_action": pair.camera_action,
                "waypoints": pair.waypoints,
            }
            with open(output_path / f"pair_{i:06d}.json", 'w') as f:
                json.dump(pair_meta, f, indent=2)

            if i % 100 == 0:
                log.info(f"Pairs generados: {len(pairs)} / {len(sequences)}")

        stats = {
            "total_sequences": len(sequences),
            "pairs_generated": len(pairs),
            "failed": failed,
            "action_distribution": action_counts,
            "avg_gap_days": np.mean([p.timestamp_gap_days for p in pairs]) if pairs else 0,
        }

        log.info(f"Cross-temporal pairing completado: {stats}")
        return stats


# ─── View Interpolator ───────────────────────────────────────────────────────

class IntermittentFreezeFrameInterpolator:
    """
    Implementa la estrategia Intermittent Freeze-Frame (Sec. 3.1 SWM).
    
    Problema: los modelos de video diffusion aprenden movimiento suave y
    continuo. Las imágenes de street-view están capturadas a 5-20m de
    intervalo → saltos bruscos.
    
    Solución: repetir cada keyframe 4 veces consecutivas para que el 3D VAE
    (con compresión temporal 4x) lo codifique correctamente.
    
    Pipeline:
        keyframes (sparse) → freeze-frame video → diffusion model → decoded video
        → discard repeated frames → smooth video
    
    Esta clase prepara el input para el modelo de interpolación
    y procesa el output para recuperar el video final.
    """

    VAE_TEMPORAL_STRIDE = 4  # Cosmos VAE comprime cada 4 frames

    def __init__(self, vae_stride: int = 4):
        self.stride = vae_stride

    def prepare_keyframe_sequence(
        self,
        keyframe_indices: list[int],
        total_frames: int,
    ) -> list[int]:
        """
        Dado un conjunto de keyframe indices, genera la secuencia
        con freeze-frames para el pipeline de interpolación.
        
        Args:
            keyframe_indices: índices de los keyframes en el video final
            total_frames: longitud total del video objetivo
        
        Returns:
            Lista de índices indicando qué frame usar en cada posición
            (keyframe = mismo índice repetido 4 veces)
        
        Example:
            keyframes=[0, 20, 40], total_frames=50
            → [0,0,0,0, 1,2,3,4, ..., 20,20,20,20, 21,..., 40,40,40,40, ...]
        """
        frame_sequence = list(range(total_frames))
        freeze_sequence = []

        kf_set = set(keyframe_indices)

        for i in range(total_frames):
            if i in kf_set:
                # Freeze: repetir 4 veces
                freeze_sequence.extend([i] * self.stride)
            else:
                freeze_sequence.append(i)

        return freeze_sequence

    def get_latent_positions(
        self,
        freeze_sequence: list[int],
        keyframe_indices: list[int],
    ) -> list[int]:
        """
        Calcula las posiciones de latentes para los keyframes
        después de la compresión VAE (4x temporal).
        
        Returns:
            Posiciones de latentes que corresponden a keyframes
        """
        kf_set = set(keyframe_indices)
        latent_positions = []
        latent_idx = 0

        i = 0
        while i < len(freeze_sequence):
            if freeze_sequence[i] in kf_set and (i == 0 or freeze_sequence[i] != freeze_sequence[i-1]):
                # Grupo de 4 frames del mismo keyframe → 1 latente
                latent_positions.append(latent_idx)
                i += self.stride
            else:
                i += 1
            latent_idx += 1

        return latent_positions

    def discard_freeze_frames(
        self,
        decoded_frames: list,
        keyframe_indices: list[int],
        freeze_positions: list[int],
    ) -> list:
        """
        Después de decodificar, descarta los frames repetidos (freeze)
        para recuperar el video final a la frecuencia correcta.
        
        Args:
            decoded_frames: frames decodificados del VAE
            keyframe_indices: índices originales de keyframes
            freeze_positions: posiciones de freeze en el video decodificado
        
        Returns:
            Video final sin frames repetidos
        """
        discard_set = set()
        kf_set = set(keyframe_indices)

        pos = 0
        for i, kf_idx in enumerate(sorted(kf_set)):
            # Las posiciones del keyframe en el video decodificado:
            # primera posición se mantiene, las 3 siguientes se descartan
            base_pos = freeze_positions[i] * self.stride
            for j in range(1, self.stride):  # descartar 3 de 4
                discard_set.add(base_pos + j)

        return [f for i, f in enumerate(decoded_frames) if i not in discard_set]

    def estimate_keyframe_positions(
        self,
        waypoints: list[tuple[float, float]],
        target_fps: float = 30.0,
        capture_interval_m: float = 10.0,
    ) -> list[int]:
        """
        Estima los índices de keyframe dado un conjunto de waypoints
        y el intervalo de captura de street-view.
        
        Args:
            waypoints: lista de (lat, lng) de las imágenes de street-view
            target_fps: FPS del video de entrenamiento
            capture_interval_m: distancia entre capturas (metros)
        
        Returns:
            Índices de frame donde van los keyframes
        """
        if len(waypoints) < 2:
            return [0]

        # Velocidad asumida: ~5 m/s (caminar) o ~14 m/s (manejar)
        # Para street-view driving asumir ~5 m/s promedio
        assumed_speed_mps = 5.0
        frames_per_capture = int(target_fps * capture_interval_m / assumed_speed_mps)

        keyframe_indices = [0]
        for i in range(1, len(waypoints)):
            keyframe_indices.append(keyframe_indices[-1] + frames_per_capture)

        return keyframe_indices


# ─── CLI / Ejemplo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-temporal pairer demo")
    parser.add_argument("--index", required=True, help="Path al geo index pickle")
    parser.add_argument("--lat", type=float, default=-33.4372)
    parser.add_argument("--lng", type=float, default=-70.6506)
    parser.add_argument("--demo", action="store_true", help="Demo con ubicaciones de Santiago")
    args = parser.parse_args()

    indexer = GeoIndexer.load(args.index)
    pairer = CrossTemporalPairer(
        indexer=indexer,
        min_timestamp_gap_days=30,
        k_references=5,
    )

    if args.demo:
        # Demo con trayectoria Alameda
        demo_waypoints = [
            (-33.4372, -70.6506),  # Plaza Italia
            (-33.4407, -70.6540),  # Bustamante
            (-33.4451, -70.6585),  # Baquedano
        ]
        action = pairer.compute_camera_action(demo_waypoints)
        print(f"Camera action: {action}")

        results = indexer.query_trajectory(demo_waypoints, k_per_point=3)
        for i, (wp, refs) in enumerate(zip(demo_waypoints, results)):
            print(f"\nWaypoint {i} ({wp[0]:.4f}, {wp[1]:.4f}): {len(refs)} refs")
            for ref, dist in refs:
                print(f"  {ref.image_id} | {dist:.0f}m | {ref.captured_date}")
    else:
        refs = indexer.query(args.lat, args.lng, k=5)
        print(f"Refs para ({args.lat}, {args.lng}):")
        for ref, dist in refs:
            print(f"  {ref.image_id} | {dist:.0f}m | heading={ref.compass_angle:.0f}°")
