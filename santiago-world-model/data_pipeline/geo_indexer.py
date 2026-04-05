"""
data_pipeline/geo_indexer.py
==============================
Indexación geoespacial de imágenes de street-view para búsqueda eficiente
de vecinos más cercanos durante el pipeline de RAG del Santiago World Model.

Implementa BallTree con métrica haversine para búsqueda en O(log n).
Compatible con el formato de salida de mapillary_scraper.py

Uso:
    indexer = GeoIndexer.from_mapillary_dir("./data/mapillary")
    indexer.build()
    indexer.save("./data/geo_index.pkl")

    # En tiempo de inferencia:
    indexer = GeoIndexer.load("./data/geo_index.pkl")
    refs = indexer.query(lat=-33.4372, lng=-70.6506, k=5, max_dist_m=200)
"""

import json
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neighbors import BallTree

log = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000


@dataclass
class StreetViewRef:
    """Referencia de street-view indexada."""
    image_id: str
    lat: float
    lng: float
    compass_angle: float
    captured_at: int
    image_path: str
    depth_path: Optional[str] = None
    camera_pose: Optional[dict] = None
    sequence_id: Optional[str] = None

    @property
    def rad_lat(self) -> float:
        return np.radians(self.lat)

    @property
    def rad_lng(self) -> float:
        return np.radians(self.lng)

    def distance_to(self, lat: float, lng: float) -> float:
        """Distancia haversine en metros a un punto."""
        dlat = np.radians(lat - self.lat)
        dlng = np.radians(lng - self.lng)
        a = (np.sin(dlat/2)**2 +
             np.cos(np.radians(self.lat)) * np.cos(np.radians(lat)) * np.sin(dlng/2)**2)
        return 2 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def heading_to(self, lat: float, lng: float) -> float:
        """Heading (bearing) en grados hacia un punto."""
        dlng = np.radians(lng - self.lng)
        x = np.sin(dlng) * np.cos(np.radians(lat))
        y = (np.cos(np.radians(self.lat)) * np.sin(np.radians(lat)) -
             np.sin(np.radians(self.lat)) * np.cos(np.radians(lat)) * np.cos(dlng))
        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360


class GeoIndexer:
    """
    Índice geoespacial para búsqueda de referencias de street-view.
    
    Usa sklearn BallTree con métrica haversine para eficiencia en
    búsquedas por radio y k-NN sobre millones de puntos.
    
    Soporta filtrado por:
    - Distancia máxima (radio en metros)
    - Diferencia de heading (ángulo de cámara)
    - Timestamp (para cross-temporal pairing)
    - Secuencia (para evitar refs de la misma captura)
    """

    def __init__(self):
        self.refs: list[StreetViewRef] = []
        self._tree: Optional[BallTree] = None
        self._coords_rad: Optional[np.ndarray] = None
        self._built = False

    @classmethod
    def from_mapillary_dir(cls, data_dir: str) -> "GeoIndexer":
        """
        Construye un indexer desde el output de mapillary_scraper.py.
        
        Args:
            data_dir: directorio con subdirs images/ y metadata/
        """
        indexer = cls()
        data_path = Path(data_dir)
        metadata_dir = data_path / "metadata"
        images_dir = data_path / "images"

        if not metadata_dir.exists():
            raise FileNotFoundError(f"No se encontró metadata en {metadata_dir}")

        meta_files = list(metadata_dir.glob("*.json"))
        log.info(f"Cargando {len(meta_files)} archivos de metadata...")

        for meta_file in meta_files:
            try:
                with open(meta_file) as f:
                    data = json.load(f)

                image_path = images_dir / f"{data['image_id']}_{data['lat']:.5f}_{data['lng']:.5f}.jpg"

                ref = StreetViewRef(
                    image_id=data["image_id"],
                    lat=data["lat"],
                    lng=data["lng"],
                    compass_angle=data.get("compass_angle", 0.0),
                    captured_at=data.get("captured_at", 0),
                    image_path=str(image_path),
                    sequence_id=data.get("sequence_id"),
                )
                indexer.refs.append(ref)

            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"Error cargando {meta_file}: {e}")

        log.info(f"Cargadas {len(indexer.refs)} referencias")
        return indexer

    @classmethod
    def from_index_file(cls, index_json: str) -> "GeoIndexer":
        """
        Construye indexer desde un archivo index_*.json de mapillary_scraper.
        Más rápido que iterar todos los JSONs individuales.
        """
        indexer = cls()
        with open(index_json) as f:
            data = json.load(f)

        for item in data:
            ref = StreetViewRef(
                image_id=item["image_id"],
                lat=item["lat"],
                lng=item["lng"],
                compass_angle=item.get("compass_angle", 0.0),
                captured_at=item.get("captured_at", 0),
                image_path=item.get("image_path", ""),
                sequence_id=item.get("sequence_id"),
            )
            indexer.refs.append(ref)

        log.info(f"Cargadas {len(indexer.refs)} referencias desde {index_json}")
        return indexer

    def build(self) -> "GeoIndexer":
        """
        Construye el BallTree. Debe llamarse antes de query().
        O(n log n) en tiempo, O(n) en espacio.
        """
        if not self.refs:
            raise ValueError("No hay referencias cargadas.")

        coords = np.array([[r.lat, r.lng] for r in self.refs])
        self._coords_rad = np.radians(coords)
        self._tree = BallTree(self._coords_rad, metric="haversine")
        self._built = True

        log.info(f"BallTree construido con {len(self.refs)} nodos")
        return self

    def query(
        self,
        lat: float,
        lng: float,
        k: int = 5,
        max_dist_m: float = 200.0,
        max_heading_diff: Optional[float] = None,
        target_heading: Optional[float] = None,
        exclude_sequence_ids: Optional[set] = None,
        exclude_timestamps_range: Optional[tuple[int, int]] = None,
    ) -> list[tuple[StreetViewRef, float]]:
        """
        Búsqueda de las k referencias más cercanas a un punto.
        
        Args:
            lat, lng: coordenadas de consulta
            k: número de referencias a retornar
            max_dist_m: radio máximo en metros
            max_heading_diff: máxima diferencia de heading (grados)
            target_heading: heading objetivo para filtrar por ángulo
            exclude_sequence_ids: excluir refs de estas secuencias
            exclude_timestamps_range: excluir refs capturadas en este rango (ms)
        
        Returns:
            Lista de (StreetViewRef, distancia_metros) ordenada por distancia
        """
        if not self._built:
            raise RuntimeError("Índice no construido. Llama build() primero.")

        query_rad = np.radians([[lat, lng]])
        radius_rad = max_dist_m / EARTH_RADIUS_M

        # BallTree query por radio
        indices, distances = self._tree.query_radius(
            query_rad,
            r=radius_rad,
            return_distance=True,
            sort_results=True
        )

        results = []
        for idx, dist_rad in zip(indices[0], distances[0]):
            ref = self.refs[idx]
            dist_m = dist_rad * EARTH_RADIUS_M

            # Filtros opcionales
            if exclude_sequence_ids and ref.sequence_id in exclude_sequence_ids:
                continue

            if exclude_timestamps_range:
                ts_min, ts_max = exclude_timestamps_range
                if ts_min <= ref.captured_at <= ts_max:
                    continue  # Cross-temporal pairing: excluir mismo período

            if max_heading_diff is not None and target_heading is not None:
                hdiff = abs(ref.compass_angle - target_heading)
                hdiff = min(hdiff, 360 - hdiff)
                if hdiff > max_heading_diff:
                    continue

            results.append((ref, dist_m))
            if len(results) >= k:
                break

        return results

    def query_trajectory(
        self,
        waypoints: list[tuple[float, float]],
        k_per_point: int = 3,
        max_dist_m: float = 150.0,
        cross_temporal_window_ms: int = 3_600_000,  # 1 hora
        current_timestamp_ms: Optional[int] = None,
    ) -> list[list[tuple[StreetViewRef, float]]]:
        """
        Recupera referencias para cada punto de una trayectoria.
        
        Implementa cross-temporal pairing: excluye referencias capturadas
        en la misma ventana temporal que el punto objetivo.
        
        Args:
            waypoints: lista de (lat, lng)
            k_per_point: referencias por waypoint
            max_dist_m: radio de búsqueda
            cross_temporal_window_ms: ventana temporal a excluir (ms)
            current_timestamp_ms: timestamp del "presente" para CT pairing
        
        Returns:
            Lista de listas de (ref, dist)
        """
        all_results = []
        exclude_range = None

        if current_timestamp_ms:
            half = cross_temporal_window_ms // 2
            exclude_range = (
                current_timestamp_ms - half,
                current_timestamp_ms + half
            )

        for lat, lng in waypoints:
            refs = self.query(
                lat=lat,
                lng=lng,
                k=k_per_point,
                max_dist_m=max_dist_m,
                exclude_timestamps_range=exclude_range,
            )
            all_results.append(refs)

        return all_results

    def get_virtual_lookahead(
        self,
        endpoint_lat: float,
        endpoint_lng: float,
        max_dist_m: float = 100.0,
    ) -> Optional[StreetViewRef]:
        """
        Recupera el Virtual Lookahead Sink para el endpoint de un chunk.
        
        Como en SWM: imagen de street-view más cercana al endpoint del chunk,
        usada como ancla futura para prevenir error accumulation.
        
        Args:
            endpoint_lat, endpoint_lng: fin del chunk actual
            max_dist_m: radio de búsqueda
        
        Returns:
            La referencia más cercana al endpoint
        """
        results = self.query(
            lat=endpoint_lat,
            lng=endpoint_lng,
            k=1,
            max_dist_m=max_dist_m,
        )
        return results[0][0] if results else None

    def coverage_stats(self, bbox: Optional[tuple] = None) -> dict:
        """
        Estadísticas de cobertura del índice.
        
        Args:
            bbox: (min_lng, min_lat, max_lng, max_lat) opcional
        """
        refs = self.refs
        if bbox:
            min_lng, min_lat, max_lng, max_lat = bbox
            refs = [r for r in refs if
                    min_lat <= r.lat <= max_lat and
                    min_lng <= r.lng <= max_lng]

        if not refs:
            return {"count": 0}

        lats = [r.lat for r in refs]
        lngs = [r.lng for r in refs]
        timestamps = [r.captured_at for r in refs]

        from datetime import datetime, timezone
        def ts_to_year(ts):
            if ts:
                return datetime.fromtimestamp(ts/1000, tz=timezone.utc).year
            return None

        years = [ts_to_year(ts) for ts in timestamps if ts]
        year_counts = {}
        for y in years:
            if y:
                year_counts[y] = year_counts.get(y, 0) + 1

        return {
            "count": len(refs),
            "lat_range": (min(lats), max(lats)),
            "lng_range": (min(lngs), max(lngs)),
            "year_distribution": year_counts,
            "sequences": len(set(r.sequence_id for r in refs if r.sequence_id)),
        }

    def save(self, path: str):
        """Serializa el índice a disco."""
        with open(path, 'wb') as f:
            pickle.dump({
                "refs": self.refs,
                "coords_rad": self._coords_rad,
                "tree": self._tree,
            }, f)
        log.info(f"Índice guardado en {path}")

    @classmethod
    def load(cls, path: str) -> "GeoIndexer":
        """Carga un índice serializado."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        indexer = cls()
        indexer.refs = data["refs"]
        indexer._coords_rad = data["coords_rad"]
        indexer._tree = data["tree"]
        indexer._built = True

        log.info(f"Índice cargado desde {path}: {len(indexer.refs)} referencias")
        return indexer


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Geo Indexer para Santiago World Model")
    subparsers = parser.add_subparsers(dest="cmd")

    # build
    p_build = subparsers.add_parser("build", help="Construir índice desde datos Mapillary")
    p_build.add_argument("--data-dir", required=True)
    p_build.add_argument("--output", default="./data/geo_index.pkl")

    # stats
    p_stats = subparsers.add_parser("stats", help="Estadísticas del índice")
    p_stats.add_argument("--index", required=True)

    # query
    p_query = subparsers.add_parser("query", help="Consulta de prueba")
    p_query.add_argument("--index", required=True)
    p_query.add_argument("--lat", type=float, required=True)
    p_query.add_argument("--lng", type=float, required=True)
    p_query.add_argument("--k", type=int, default=5)
    p_query.add_argument("--radius", type=float, default=200.0)

    args = parser.parse_args()

    if args.cmd == "build":
        indexer = GeoIndexer.from_mapillary_dir(args.data_dir)
        indexer.build()
        indexer.save(args.output)
        stats = indexer.coverage_stats()
        print(json.dumps(stats, indent=2))

    elif args.cmd == "stats":
        indexer = GeoIndexer.load(args.index)
        print(json.dumps(indexer.coverage_stats(), indent=2))

    elif args.cmd == "query":
        indexer = GeoIndexer.load(args.index)
        results = indexer.query(args.lat, args.lng, k=args.k, max_dist_m=args.radius)
        for ref, dist in results:
            print(f"  {ref.image_id} | dist={dist:.1f}m | heading={ref.compass_angle:.1f}° | {ref.captured_at}")

    else:
        parser.print_help()
