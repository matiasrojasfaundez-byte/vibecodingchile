"""
data_pipeline/mapillary_scraper.py
===================================
Descarga imágenes de Mapillary API v4 geoindexadas para Santiago de Chile.
Produce un dataset de pares (imagen, metadata) listos para el pipeline SWM.

Uso:
    python mapillary_scraper.py --token YOUR_TOKEN --output ./data/mapillary
    python mapillary_scraper.py --token YOUR_TOKEN --bbox santiago_centro --limit 5000

Obtener token: https://www.mapillary.com/developer/api-documentation
"""

import os
import time
import json
import hashlib
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from PIL import Image
from io import BytesIO

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ─── Santiago bounding boxes ────────────────────────────────────────────────

SANTIAGO_BBOXES = {
    "full_rm": (-70.8500, -33.6500, -70.4500, -33.3000),        # Región Metropolitana completa
    "santiago_centro": (-70.7000, -33.5000, -70.6000, -33.3900), # Santiago centro
    "providencia": (-70.6200, -33.4400, -70.5900, -33.4100),
    "las_condes": (-70.6000, -33.4300, -70.5400, -33.3900),
    "nunoa": (-70.6200, -33.4800, -70.5800, -33.4400),
    "maipu": (-70.7800, -33.5200, -70.7200, -33.4800),
    "barrio_republica": (-70.6750, -33.4650, -70.6550, -33.4450),
    "alameda": (-70.6700, -33.4550, -70.6400, -33.4300),
}


@dataclass
class MapillaryImage:
    """Imagen de Mapillary con metadata completa."""
    image_id: str
    lat: float
    lng: float
    compass_angle: float          # heading en grados [0, 360)
    captured_at: int              # Unix timestamp ms
    thumb_256_url: str
    thumb_1024_url: str
    thumb_2048_url: str
    sequence_id: Optional[str]
    is_pano: bool
    creator_id: Optional[str]
    altitude: Optional[float]
    camera_type: Optional[str]

    @property
    def captured_date(self) -> str:
        """Fecha de captura como string ISO."""
        from datetime import datetime, timezone
        return datetime.fromtimestamp(
            self.captured_at / 1000, tz=timezone.utc
        ).strftime('%Y-%m-%d')

    @property
    def local_filename(self) -> str:
        """Nombre de archivo local único."""
        return f"{self.image_id}_{self.lat:.5f}_{self.lng:.5f}.jpg"


class MapillaryScraper:
    """
    Scraper para la API v4 de Mapillary.
    
    Implementa:
    - Paginación automática por bbox
    - Filtrado por fecha y secuencia
    - Descarga paralela de imágenes
    - Rate limiting respetuoso
    - Cache local para evitar re-descargas
    """

    BASE_URL = "https://graph.mapillary.com"
    IMAGES_ENDPOINT = "/images"
    RATE_LIMIT_SLEEP = 0.1   # segundos entre requests
    MAX_RETRIES = 3
    RETRY_SLEEP = 2.0

    # Campos a solicitar en la API
    FIELDS = ",".join([
        "id",
        "geometry",
        "thumb_256_url",
        "thumb_1024_url",
        "thumb_2048_url",
        "captured_at",
        "compass_angle",
        "sequence",
        "is_pano",
        "creator",
        "altitude",
        "camera_type",
    ])

    def __init__(self, access_token: str, output_dir: str = "./data/mapillary"):
        self.token = access_token
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"OAuth {self.token}",
            "Content-Type": "application/json",
        })

        self._downloaded = set(self._load_existing_ids())
        log.info(f"Scraper inicializado. {len(self._downloaded)} imágenes ya descargadas.")

    def _load_existing_ids(self) -> list:
        """Carga IDs ya descargados para evitar re-descarga."""
        index_file = self.output_dir / "downloaded_ids.txt"
        if index_file.exists():
            return index_file.read_text().strip().splitlines()
        return []

    def _save_id(self, image_id: str):
        index_file = self.output_dir / "downloaded_ids.txt"
        with open(index_file, 'a') as f:
            f.write(image_id + '\n')
        self._downloaded.add(image_id)

    def _request(self, params: dict) -> dict:
        """Request con retry y rate limiting."""
        url = f"{self.BASE_URL}{self.IMAGES_ENDPOINT}"
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                time.sleep(self.RATE_LIMIT_SLEEP)
                return resp.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait = self.RETRY_SLEEP * (attempt + 1) * 5
                    log.warning(f"Rate limited. Esperando {wait}s...")
                    time.sleep(wait)
                elif e.response.status_code == 401:
                    raise ValueError("Token de Mapillary inválido o expirado.") from e
                else:
                    log.error(f"HTTP {e.response.status_code}: {e}")
                    if attempt == self.MAX_RETRIES - 1:
                        raise
            except requests.exceptions.RequestException as e:
                log.error(f"Request error (intento {attempt+1}): {e}")
                if attempt == self.MAX_RETRIES - 1:
                    raise
                time.sleep(self.RETRY_SLEEP)
        return {}

    def iter_bbox(
        self,
        bbox: tuple[float, float, float, float],  # min_lng, min_lat, max_lng, max_lat
        limit_per_page: int = 200,
        max_total: Optional[int] = None,
        min_captured: Optional[str] = None,  # "YYYY-MM-DD"
        max_captured: Optional[str] = None,
        pano_only: bool = False,
    ) -> Iterator[MapillaryImage]:
        """
        Itera sobre todas las imágenes en un bounding box.
        Maneja paginación automáticamente.
        
        Args:
            bbox: (min_lng, min_lat, max_lng, max_lat)
            limit_per_page: imágenes por página (max 200 en API)
            max_total: límite total de imágenes
            min_captured: fecha mínima ISO
            max_captured: fecha máxima ISO
            pano_only: solo imágenes panorámicas (360°)
        
        Yields:
            MapillaryImage con metadata completa
        """
        min_lng, min_lat, max_lng, max_lat = bbox
        bbox_str = f"{min_lng},{min_lat},{max_lng},{max_lat}"

        params = {
            "fields": self.FIELDS,
            "bbox": bbox_str,
            "limit": min(limit_per_page, 200),
        }

        if pano_only:
            params["is_pano"] = "true"

        total = 0
        page = 0

        while True:
            data = self._request(params)
            if not data or "data" not in data:
                break

            features = data["data"]
            if not features:
                break

            for feat in features:
                if max_total and total >= max_total:
                    return

                img = self._parse_image(feat)
                if img is None:
                    continue

                # Filtrar por fecha
                if min_captured and img.captured_date < min_captured:
                    continue
                if max_captured and img.captured_date > max_captured:
                    continue

                total += 1
                yield img

            page += 1
            log.debug(f"Página {page}: {len(features)} imágenes, total={total}")

            # Paginación con cursor
            paging = data.get("paging", {})
            next_url = paging.get("next")
            if not next_url:
                break

            # Extraer cursor del next URL
            import urllib.parse as up
            parsed = up.urlparse(next_url)
            qs = up.parse_qs(parsed.query)
            if "after" in qs:
                params["after"] = qs["after"][0]
            else:
                break

        log.info(f"Iteración completada: {total} imágenes en bbox")

    def _parse_image(self, feat: dict) -> Optional[MapillaryImage]:
        """Parsea un feature de la API a MapillaryImage."""
        try:
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [None, None])
            if not coords[0]:
                return None

            return MapillaryImage(
                image_id=feat["id"],
                lat=coords[1],
                lng=coords[0],
                compass_angle=feat.get("compass_angle", 0.0),
                captured_at=feat.get("captured_at", 0),
                thumb_256_url=feat.get("thumb_256_url", ""),
                thumb_1024_url=feat.get("thumb_1024_url", ""),
                thumb_2048_url=feat.get("thumb_2048_url", ""),
                sequence_id=feat.get("sequence"),
                is_pano=feat.get("is_pano", False),
                creator_id=feat.get("creator", {}).get("id"),
                altitude=feat.get("altitude"),
                camera_type=feat.get("camera_type"),
            )
        except (KeyError, TypeError, IndexError) as e:
            log.warning(f"Error parseando imagen: {e}")
            return None

    def download_image(self, img: MapillaryImage, resolution: str = "1024") -> Optional[Path]:
        """
        Descarga una imagen a disco.
        
        Args:
            img: MapillaryImage
            resolution: "256" | "1024" | "2048"
        
        Returns:
            Path al archivo guardado, o None si falla
        """
        if img.image_id in self._downloaded:
            return self.images_dir / img.local_filename

        url_map = {
            "256": img.thumb_256_url,
            "1024": img.thumb_1024_url,
            "2048": img.thumb_2048_url,
        }
        url = url_map.get(resolution, img.thumb_1024_url)

        if not url:
            log.warning(f"Sin URL para imagen {img.image_id} a resolución {resolution}")
            return None

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            img_data = Image.open(BytesIO(resp.content)).convert("RGB")
            out_path = self.images_dir / img.local_filename
            img_data.save(out_path, "JPEG", quality=90)

            # Guardar metadata
            meta_path = self.metadata_dir / f"{img.image_id}.json"
            with open(meta_path, 'w') as f:
                json.dump(asdict(img), f, indent=2)

            self._save_id(img.image_id)
            return out_path

        except Exception as e:
            log.error(f"Error descargando {img.image_id}: {e}")
            return None

    def scrape_region(
        self,
        region: str = "santiago_centro",
        max_images: int = 10000,
        resolution: str = "1024",
        num_workers: int = 8,
        **kwargs,
    ) -> dict:
        """
        Pipeline completo para scraping de una región predefinida.
        
        Args:
            region: clave de SANTIAGO_BBOXES
            max_images: máximo de imágenes a descargar
            resolution: resolución de descarga
            num_workers: workers paralelos para descarga
        
        Returns:
            dict con estadísticas del scraping
        """
        if region not in SANTIAGO_BBOXES:
            raise ValueError(f"Región '{region}' no conocida. Disponibles: {list(SANTIAGO_BBOXES.keys())}")

        bbox = SANTIAGO_BBOXES[region]
        log.info(f"Iniciando scraping: {region} | bbox={bbox} | max={max_images}")

        images_to_download = []
        metadata_index = []

        for img in self.iter_bbox(bbox, max_total=max_images, **kwargs):
            images_to_download.append(img)
            metadata_index.append(asdict(img))

            if len(images_to_download) % 500 == 0:
                log.info(f"Metadatos recopilados: {len(images_to_download)}")

        # Guardar índice de metadata completo
        index_path = self.output_dir / f"index_{region}.json"
        with open(index_path, 'w') as f:
            json.dump(metadata_index, f, indent=2)
        log.info(f"Índice guardado: {index_path} ({len(metadata_index)} imágenes)")

        # Descarga paralela
        log.info(f"Descargando {len(images_to_download)} imágenes con {num_workers} workers...")
        downloaded = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.download_image, img, resolution): img
                for img in images_to_download
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    downloaded += 1
                else:
                    failed += 1

                if (downloaded + failed) % 100 == 0:
                    log.info(f"Progreso: {downloaded} ok / {failed} failed")

        stats = {
            "region": region,
            "total_metadata": len(metadata_index),
            "downloaded": downloaded,
            "failed": failed,
            "output_dir": str(self.output_dir),
        }

        log.info(f"Scraping completado: {stats}")
        return stats

    def scrape_custom_bbox(
        self,
        min_lng: float, min_lat: float,
        max_lng: float, max_lat: float,
        **kwargs
    ) -> dict:
        """Scraping con bbox custom."""
        bbox = (min_lng, min_lat, max_lng, max_lat)
        images = list(self.iter_bbox(bbox, **kwargs))
        log.info(f"Custom bbox: {len(images)} imágenes encontradas")
        return {"images": [asdict(i) for i in images]}


def main():
    parser = argparse.ArgumentParser(
        description="Mapillary scraper para Santiago World Model"
    )
    parser.add_argument("--token", required=True, help="Mapillary access token")
    parser.add_argument("--output", default="./data/mapillary", help="Directorio de salida")
    parser.add_argument(
        "--region", default="santiago_centro",
        choices=list(SANTIAGO_BBOXES.keys()),
        help="Región a scraping"
    )
    parser.add_argument("--limit", type=int, default=10000, help="Máximo de imágenes")
    parser.add_argument("--resolution", choices=["256", "1024", "2048"], default="1024")
    parser.add_argument("--workers", type=int, default=8, help="Workers paralelos")
    parser.add_argument("--pano-only", action="store_true", help="Solo imágenes 360°")
    parser.add_argument(
        "--bbox", nargs=4, type=float,
        metavar=("MIN_LNG", "MIN_LAT", "MAX_LNG", "MAX_LAT"),
        help="Bbox custom en vez de región predefinida"
    )
    args = parser.parse_args()

    scraper = MapillaryScraper(
        access_token=args.token,
        output_dir=args.output,
    )

    if args.bbox:
        result = scraper.scrape_custom_bbox(
            *args.bbox,
            max_total=args.limit,
            pano_only=args.pano_only,
        )
    else:
        result = scraper.scrape_region(
            region=args.region,
            max_images=args.limit,
            resolution=args.resolution,
            num_workers=args.workers,
            pano_only=args.pano_only,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
