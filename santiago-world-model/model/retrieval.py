"""
model/retrieval.py
===================
Pipeline de RAG (Retrieval-Augmented Generation) para Santiago World Model.
Implementa las Secciones 4.1, 4.2 y 4.3 del paper SWM.

Componentes:
- StreetViewRetrieval: recupera referencias por GPS + filtrado por profundidad
- GeometricReferencing: warping depth-based de referencia a viewpoint objetivo
- SemanticReferencing: prepara tokens de referencia para el DiT transformer
- VirtualLookaheadSink: ancla futura para prevenir error accumulation
"""

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)


# ─── Estructuras de datos ────────────────────────────────────────────────────

@dataclass
class CameraPose:
    """Pose de cámara 6-DoF."""
    rotation: np.ndarray       # (3, 3) rotation matrix
    translation: np.ndarray    # (3,) translation vector
    intrinsics: np.ndarray     # (3, 3) K matrix (fx, fy, cx, cy, ...)
    image_width: int
    image_height: int

    @property
    def extrinsics(self) -> np.ndarray:
        """Matrix 4x4 [R|t]."""
        E = np.eye(4)
        E[:3, :3] = self.rotation
        E[:3, 3] = self.translation
        return E

    def to_plucker_rays(self) -> np.ndarray:
        """
        Calcula Plücker ray embeddings (6 canales) para cada pixel.
        Usado por el DiT para codificar poses de cámara.
        
        Returns: (H, W, 6) array de Plücker rays
        """
        H, W = self.image_height, self.image_width
        K_inv = np.linalg.inv(self.intrinsics)
        R = self.rotation
        t = self.translation

        # Grid de pixels
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        ones = np.ones_like(uu)
        pixel_coords = np.stack([uu, vv, ones], axis=-1).reshape(-1, 3)

        # Direcciones de rayo en coordenadas de cámara
        dirs_cam = (K_inv @ pixel_coords.T).T  # (HW, 3)

        # Rotar a coordenadas del mundo
        dirs_world = (R @ dirs_cam.T).T  # (HW, 3)
        dirs_world = dirs_world / (np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-8)

        # Origen del rayo: posición de la cámara en mundo
        cam_pos = -R.T @ t  # (3,)
        origins = np.broadcast_to(cam_pos, dirs_world.shape)

        # Plücker = (d, o × d)
        moments = np.cross(origins, dirs_world)  # (HW, 3)
        plucker = np.concatenate([dirs_world, moments], axis=-1)  # (HW, 6)

        return plucker.reshape(H, W, 6)


@dataclass
class RetrievedReference:
    """Referencia recuperada con imagen, pose y profundidad."""
    image_id: str
    image: np.ndarray           # (H, W, 3) RGB float [0,1]
    depth: np.ndarray           # (H, W) depth map en metros
    pose: CameraPose
    distance_m: float           # distancia al punto de query
    plucker_rays: Optional[np.ndarray] = None  # (H, W, 6)

    def __post_init__(self):
        if self.plucker_rays is None and self.pose is not None:
            self.plucker_rays = self.pose.to_plucker_rays()


# ─── Street-View Retrieval ───────────────────────────────────────────────────

class StreetViewRetrieval:
    """
    Implementa la Sec. 4.1 del paper SWM.
    
    Pipeline de recuperación en dos etapas:
    1. Nearest-neighbor search por GPS → candidatos
    2. Depth-based reprojection filtering → filtrar por cobertura
    """

    def __init__(self, indexer, depth_model=None):
        """
        Args:
            indexer: GeoIndexer con las referencias indexadas
            depth_model: modelo de depth estimation (Depth Anything V3)
        """
        self.indexer = indexer
        self.depth_model = depth_model

    def retrieve_for_chunk(
        self,
        target_trajectory: list[tuple[float, float]],
        k: int = 5,
        max_dist_m: float = 150.0,
        coverage_threshold: float = 0.1,
    ) -> list[RetrievedReference]:
        """
        Recupera referencias para un chunk de generación.
        
        Stage 1: Nearest-neighbor search por GPS
        Stage 2: Depth-based reprojection filtering
        
        Args:
            target_trajectory: [(lat, lng)] del chunk actual
            k: número de referencias a retornar
            max_dist_m: radio de búsqueda
            coverage_threshold: fracción mínima de pixels proyectados
        
        Returns:
            Lista de RetrievedReference ordenada por distancia
        """
        candidates = []

        # Stage 1: NN search
        for lat, lng in target_trajectory:
            results = self.indexer.query(
                lat=lat, lng=lng,
                k=k * 2,
                max_dist_m=max_dist_m,
            )
            candidates.extend(results)

        # Deduplicar
        seen = set()
        unique_candidates = []
        for ref, dist in candidates:
            if ref.image_id not in seen:
                seen.add(ref.image_id)
                unique_candidates.append((ref, dist))

        unique_candidates.sort(key=lambda x: x[1])

        # Stage 2: Depth-based reprojection filtering
        # En la demo, simplificamos — en producción usaría el depth model
        # para verificar cobertura de proyección
        filtered = self._depth_reprojection_filter(
            unique_candidates,
            target_trajectory,
            coverage_threshold,
        )

        return filtered[:k]

    def _depth_reprojection_filter(
        self,
        candidates: list,
        target_trajectory: list[tuple[float, float]],
        coverage_threshold: float,
    ) -> list[RetrievedReference]:
        """
        Filtra referencias verificando que proyectan suficientes pixels
        en el viewpoint objetivo.
        
        En producción: usa Depth Anything V3 para depth maps reales.
        Aquí: filtro heurístico basado en distancia y heading.
        """
        refs = []

        # Centro de la trayectoria
        center_lat = np.mean([wp[0] for wp in target_trajectory])
        center_lng = np.mean([wp[1] for wp in target_trajectory])

        for ref, dist in candidates:
            # Heurística: refs cercanas con heading compatible tienen buena cobertura
            heading_to_center = ref.heading_to(center_lat, center_lng)
            heading_diff = abs(ref.compass_angle - heading_to_center)
            heading_diff = min(heading_diff, 360 - heading_diff)

            # Coverage estimada: mayor para refs cercanas y con heading compatible
            coverage = max(0, 1.0 - dist / 200.0) * max(0, 1.0 - heading_diff / 90.0)

            if coverage >= coverage_threshold:
                # En producción: cargar imagen y depth map reales
                mock_ref = RetrievedReference(
                    image_id=ref.image_id,
                    image=np.zeros((256, 256, 3), dtype=np.float32),  # placeholder
                    depth=np.ones((256, 256), dtype=np.float32),       # placeholder
                    pose=None,  # sería cargado del disco
                    distance_m=dist,
                )
                refs.append(mock_ref)

        return refs


# ─── Geometric Referencing ───────────────────────────────────────────────────

class GeometricReferencing:
    """
    Implementa Geometric Referencing (Sec. 4.3 SWM).
    
    Para cada frame objetivo, reprojecta la referencia más cercana
    al viewpoint objetivo usando depth-based forward splatting.
    
    ref_image + ref_depth + relative_pose → warped_image
    """

    def forward_splat(
        self,
        src_image: np.ndarray,   # (H, W, 3)
        src_depth: np.ndarray,   # (H, W)
        src_pose: CameraPose,
        tgt_pose: CameraPose,
    ) -> np.ndarray:
        """
        Depth-based forward splatting: reprojecta src_image al viewpoint tgt.
        
        Implementa Eq. (2) del paper:
            x_warp,t = Render(Unproj(x_ref, d_ref), c_ref→t)
        
        Args:
            src_image: imagen fuente (referencia)
            src_depth: mapa de profundidad de la fuente
            src_pose: pose de la cámara fuente
            tgt_pose: pose de la cámara objetivo
        
        Returns:
            Imagen warpeada en el viewpoint objetivo (H, W, 3)
        """
        H, W = src_depth.shape

        # 1. Unproyectar: imagen 2D → puntos 3D usando depth
        K_src = src_pose.intrinsics
        R_src = src_pose.rotation
        t_src = src_pose.translation

        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)

        # Coordenadas homogéneas de pixel
        px = (uu - K_src[0, 2]) / K_src[0, 0]
        py = (vv - K_src[1, 2]) / K_src[1, 1]

        # Puntos 3D en coordenadas de cámara src
        pts_cam_src = np.stack([
            px * src_depth,
            py * src_depth,
            src_depth
        ], axis=-1)  # (H, W, 3)

        # Transformar a mundo
        cam_pos_src = -R_src.T @ t_src
        pts_world = (R_src.T @ (pts_cam_src.reshape(-1, 3) - t_src).T).T
        pts_world = pts_world.reshape(H, W, 3)

        # 2. Render: puntos 3D → imagen tgt usando pose objetivo
        K_tgt = tgt_pose.intrinsics
        R_tgt = tgt_pose.rotation
        t_tgt = tgt_pose.translation

        # Proyectar al viewpoint objetivo
        pts_cam_tgt = (R_tgt @ pts_world.reshape(-1, 3).T + t_tgt[:, None]).T
        pts_cam_tgt = pts_cam_tgt.reshape(H, W, 3)

        z_tgt = pts_cam_tgt[:, :, 2]
        valid = z_tgt > 0

        u_tgt = np.where(valid, (pts_cam_tgt[:, :, 0] / z_tgt) * K_tgt[0, 0] + K_tgt[0, 2], -1)
        v_tgt = np.where(valid, (pts_cam_tgt[:, :, 1] / z_tgt) * K_tgt[1, 1] + K_tgt[1, 2], -1)

        # 3. Splat: scatter pixels fuente a posiciones objetivo
        warped = np.zeros((H, W, 3), dtype=np.float32)
        z_buffer = np.full((H, W), np.inf)

        u_int = np.clip(u_tgt.astype(int), 0, W-1)
        v_int = np.clip(v_tgt.astype(int), 0, H-1)

        # Rasterización simple (sin anti-aliasing)
        mask = valid.flatten()
        src_flat = src_image.reshape(-1, 3)
        u_flat = u_int.flatten()
        v_flat = v_int.flatten()
        z_flat = z_tgt.flatten()

        for i in range(len(mask)):
            if not mask[i]:
                continue
            vi, ui = v_flat[i], u_flat[i]
            if z_flat[i] < z_buffer[vi, ui]:
                z_buffer[vi, ui] = z_flat[i]
                warped[vi, ui] = src_flat[i]

        return warped


# ─── Virtual Lookahead Sink ──────────────────────────────────────────────────

class VirtualLookaheadSink:
    """
    Implementa el Virtual Lookahead Sink (Sec. 4.2 del paper SWM).
    
    Problema: la generación autoregresiva acumula error. El attention sink
    clásico (primer frame) se vuelve irrelevante cuando la cámara se aleja.
    
    Solución: recuperar dinámicamente una imagen de street-view cercana al
    ENDPOINT del chunk actual y colocarla como frame "futuro virtual".
    Esto da al modelo un ancla limpia y relevante espacialmente.
    
    Implementación:
    - z_VL: latente de la imagen de lookahead
    - RoPE position: H + L + Δ_VL (más allá del chunk actual)
    - El modelo aprende a "converger" hacia este ancla
    
    Z_seq = [Z_hist; Z_target; z_VL]
    p_seq = [1..H; H+1..H+L; H+L+Δ_VL]
    """

    def __init__(
        self,
        indexer,
        delta_vl: int = 5,        # Offset temporal en posiciones RoPE
        max_dist_m: float = 100.0,
    ):
        """
        Args:
            indexer: GeoIndexer para recuperar el lookahead
            delta_vl: hyperparámetro Δ_VL (offset de posición RoPE)
            max_dist_m: radio de búsqueda del lookahead
        """
        self.indexer = indexer
        self.delta_vl = delta_vl
        self.max_dist_m = max_dist_m

    def get_sink_image(
        self,
        chunk_endpoint_lat: float,
        chunk_endpoint_lng: float,
    ):
        """
        Recupera la imagen de street-view más cercana al endpoint del chunk.
        
        Esta imagen actúa como "destino virtual futuro" para estabilizar
        la generación a largo plazo.
        
        Args:
            chunk_endpoint_lat, chunk_endpoint_lng: fin del chunk actual
        
        Returns:
            StreetViewRef de la imagen lookahead, o None si no hay disponible
        """
        sink = self.indexer.get_virtual_lookahead(
            endpoint_lat=chunk_endpoint_lat,
            endpoint_lng=chunk_endpoint_lng,
            max_dist_m=self.max_dist_m,
        )

        if sink is None:
            log.warning(
                f"No se encontró VL Sink para ({chunk_endpoint_lat:.4f}, {chunk_endpoint_lng:.4f}). "
                "Generación continúa sin ancla."
            )

        return sink

    def build_token_sequence(
        self,
        history_latents: np.ndarray,   # (H, latent_dim)
        target_latents: np.ndarray,    # (L, latent_dim)
        vl_latent: Optional[np.ndarray],  # (1, latent_dim)
    ) -> tuple[np.ndarray, list[int]]:
        """
        Construye la secuencia de tokens con posiciones RoPE.
        Implementa Eq. (1) del paper.
        
        Z_seq = [Z_hist; Z_target; z_VL]
        p_seq = [1..H; H+1..H+L; H+L+Δ_VL]
        
        Args:
            history_latents: H latentes de historia
            target_latents: L latentes del chunk actual
            vl_latent: latente del VL sink (None si no disponible)
        
        Returns:
            (token_sequence, rope_positions)
        """
        H = len(history_latents)
        L = len(target_latents)

        if vl_latent is not None:
            token_sequence = np.concatenate([
                history_latents,
                target_latents,
                vl_latent,
            ], axis=0)  # (H + L + 1, latent_dim)

            rope_positions = (
                list(range(1, H + 1)) +          # historia: 1..H
                list(range(H + 1, H + L + 1)) +  # target: H+1..H+L
                [H + L + self.delta_vl]           # sink: H+L+Δ
            )
        else:
            # Sin sink: solo historia + target
            token_sequence = np.concatenate([history_latents, target_latents], axis=0)
            rope_positions = list(range(1, H + L + 1))

        return token_sequence, rope_positions

    def train_sink_loss_weight(self, delta_vl_actual: int) -> float:
        """
        Durante training, el sink usa un frame futuro real a distancia
        aleatoria delta_vl_actual. El peso de loss se calcula según
        la proximidad del ancla.
        
        Como en el paper: exponer el modelo a distintas distancias
        de lookahead para que aprenda a usarlo a cualquier distancia.
        """
        # Mayor peso para sinks más cercanos (más informativos)
        return np.exp(-delta_vl_actual / 10.0)
