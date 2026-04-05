# Santiago World Model (SWM-SCL)

> *¿Qué pasaría si pudieras recorrer virtualmente La Alameda, ver un tsunami inundar Plaza Italia, o navegar el barrio República como en un videojuego? Santiago World Model lleva la simulación de mundo al mundo real.*

**Santiago World Model (SWM-SCL)** es una adaptación del [Seoul World Model](https://seoul-world-model.github.io) (NAVER AI Lab / KAIST) para Santiago de Chile. Genera video foto-realista navegable anclado a las calles reales de Santiago mediante **Retrieval-Augmented Generation** sobre imágenes de street-view.

[![Demo](https://img.shields.io/badge/Demo-Interactiva-00ff88?style=flat-square)](./demo/index.html)
[![Paper SWM](https://img.shields.io/badge/Paper-arXiv:2603.15583-cyan?style=flat-square)](https://arxiv.org/abs/2603.15583)
[![VibeCodingChile](https://img.shields.io/badge/by-VibeCodingChile-red?style=flat-square)](https://vibecodingchile.cl)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)]()

---

## Demo Interactiva

Abre `demo/index.html` en tu navegador (o hostea en GitHub Pages):

- **3 rutas pre-cargadas**: La Alameda, Barrio República, Providencia
- **Mapa real de Santiago** con Leaflet + CartoDB dark tiles
- **Visualización del pipeline RAG**: retrieve → geometric ref → semantic ref → VL sink → DiT → decode
- **Click en el mapa** para agregar waypoints custom

> ⚠️ La demo simula el pipeline. El modelo generativo real requiere fine-tune del base model (ver [Sección Entrenamiento](#entrenamiento)).

---

## Qué es esto

SWM-SCL es un **video world model anclado en el mundo real**. A diferencia de los world models clásicos que "imaginan" todo, este modelo:

1. Toma coordenadas GPS + trayectoria de cámara + prompt de texto
2. **Recupera imágenes de street-view** reales de Santiago cercanas al punto (RAG)
3. Las usa como referencias geométricas y de apariencia para **generar video foto-realista**
4. El video resultante muestra las calles reales de Santiago, no un entorno imaginado

```
GPS + Trayectoria + Texto
        ↓
  Retrieval (Mapillary)
        ↓
  Geometric Ref (depth splatting) ──┐
  Semantic Ref (token injection)    ├→ DiT (Cosmos-2B) → Video
  Virtual Lookahead Sink ───────────┘
```

---

## Arquitectura

### 1. Cross-Temporal Pairing

Las referencias de street-view deben venir de **timestamps distintos** al video objetivo. Esto fuerza al modelo a aprender estructura persistente (edificios, calles) e ignorar contenido dinámico (autos, peatones).

Sin cross-temporal pairing → el modelo copia vehículos de la referencia  
Con cross-temporal pairing → el modelo enfoca atención en geometría urbana

### 2. Virtual Lookahead Sink

El problema de los world models autoregressivos: acumulan error. El attention sink clásico (primer frame) se vuelve irrelevante.

Solución: recuperar dinámicamente la imagen de street-view más cercana al **endpoint del chunk actual** y colocarla como "destino futuro virtual":

```
Z_seq = [Z_hist ; Z_target ; z_VL]
p_seq = [1..H   ; H+1..H+L ; H+L+Δ_VL]
```

Esto re-ancla la generación a la geometría real en cada paso.

### 3. Intermittent Freeze-Frame

El 3D VAE comprime cada 4 frames en 1 latente. Las imágenes de street-view están a 5-20m de intervalo → saltos bruscos en el video.

Solución: repetir cada keyframe 4 veces antes de codificar, luego descartar las repeticiones al decodificar. Sin modificar la arquitectura del modelo.

### 4. Geometric + Semantic Referencing

**Geometric**: depth-based forward splatting de la referencia al viewpoint objetivo. Provee layout espacial.

**Semantic**: inyección de tokens de referencia en el DiT transformer. Preserva detalles de apariencia.

Ambos son complementarios y necesarios (ablation en el paper original).

---

## Estructura del Proyecto

```
santiago-world-model/
├── demo/
│   └── index.html              # Demo interactiva (GitHub Pages ready)
├── data_pipeline/
│   ├── mapillary_scraper.py    # Descarga imágenes de Mapillary API v4
│   ├── geo_indexer.py          # BallTree geoespacial para RAG
│   ├── cross_temporal_pairer.py # Cross-temporal pairing + view interpolation
│   └── caption_generator.py    # Generación de captions con VLM
├── model/
│   ├── retrieval.py            # Pipeline RAG: retrieve + geometric + semantic
│   └── vl_sink.py              # Virtual Lookahead Sink (standalone)
├── configs/
│   └── santiago_config.yaml    # Configuración completa del experimento
├── scripts/
│   ├── build_index.sh          # Build geo index desde datos scrapeados
│   └── run_demo.sh             # Levantar demo local
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
git clone https://github.com/vibecodingchile/santiago-world-model
cd santiago-world-model
pip install -r requirements.txt
```

Para la demo interactiva (solo HTML, sin dependencias Python):
```bash
open demo/index.html
# o
python -m http.server 8080  # y abrir localhost:8080/demo/
```

---

## Pipeline de Datos

### 1. Obtener token Mapillary

Regístrate en [mapillary.com/developer](https://www.mapillary.com/developer/api-documentation) y crea un token de acceso.

```bash
export MAPILLARY_TOKEN=tu_token_aqui
```

### 2. Scraping de imágenes

```bash
# Santiago centro (~10K imágenes, ~15 min)
python -m data_pipeline.mapillary_scraper \
  --token $MAPILLARY_TOKEN \
  --region santiago_centro \
  --limit 10000 \
  --output ./data/mapillary

# Cobertura completa Santiago RM (~200K imágenes, varias horas)
python -m data_pipeline.mapillary_scraper \
  --token $MAPILLARY_TOKEN \
  --region full_rm \
  --limit 200000 \
  --workers 16 \
  --output ./data/mapillary
```

Regiones disponibles:
- `full_rm` — Región Metropolitana completa
- `santiago_centro` — Santiago centro
- `providencia`, `las_condes`, `nunoa`, `maipu`
- `barrio_republica` — Barrio República
- `alameda` — Av. Libertador Bernardo O'Higgins

### 3. Construir índice geoespacial

```bash
python -m data_pipeline.geo_indexer build \
  --data-dir ./data/mapillary \
  --output ./data/geo_index.pkl
```

Estadísticas del índice:
```bash
python -m data_pipeline.geo_indexer stats --index ./data/geo_index.pkl
```

### 4. Test de retrieval

```bash
# Test en Plaza Italia
python -m data_pipeline.geo_indexer query \
  --index ./data/geo_index.pkl \
  --lat -33.4372 --lng -70.6506 \
  --k 5 --radius 200
```

### 5. Generar pares de entrenamiento

```bash
python -m data_pipeline.cross_temporal_pairer \
  --index ./data/geo_index.pkl \
  --demo
```

---

## Entrenamiento

> ⚠️ El entrenamiento completo requiere infraestructura significativa (24x H100 GPUs en el paper original). Para experimentación, considera fine-tune sobre un subset pequeño o usar LoRA.

### Prerequisitos

1. Descarga el modelo base [Cosmos-Predict2.5-2B](https://github.com/NVIDIA/Cosmos) de NVIDIA
2. Prepara el dataset con el pipeline de datos (ver arriba)
3. Ajusta `configs/santiago_config.yaml` según tus recursos

### Fine-tune (Teacher Forcing)

```bash
python scripts/train.py \
  --config configs/santiago_config.yaml \
  --mode teacher_forcing \
  --iterations 10000 \
  --batch_size 4 \
  --gpus 1
```

### Self-Forcing (para inference en tiempo real)

```bash
python scripts/train.py \
  --config configs/santiago_config.yaml \
  --mode self_forcing \
  --checkpoint checkpoints/santiago_wm/tf_best.ckpt
```

---

## Inferencia

```bash
# Generar video de La Alameda
python scripts/generate.py \
  --checkpoint checkpoints/santiago_wm/best.ckpt \
  --start-lat -33.4372 \
  --start-lng -70.6506 \
  --trajectory alameda \
  --prompt "A sunny afternoon on La Alameda, busy traffic, Andes mountains in the background" \
  --output output/alameda.mp4

# Con texto imaginativo
python scripts/generate.py \
  --checkpoint checkpoints/santiago_wm/best.ckpt \
  --start-lat -33.4513 \
  --start-lng -70.6653 \
  --trajectory plaza_italia \
  --prompt "A massive earthquake cracks the streets of Santiago" \
  --output output/terremoto.mp4
```

---

## Comparación con Seoul World Model

| Feature | Seoul WM (NAVER) | Santiago WM (este repo) |
|---|---|---|
| Ciudad | Seúl, Corea | Santiago, Chile |
| Imágenes | 1.2M NAVER Map | Mapillary (open) |
| Modelo base | Cosmos-Predict2.5-2B | Cosmos-Predict2.5-2B |
| Datos sintéticos | CARLA (431K m²) | CARLA (WIP) |
| Cross-temporal pairing | ✅ | ✅ |
| VL Sink | ✅ | ✅ |
| Demo interactiva | Sitio web | `demo/index.html` |
| Checkpoints públicos | Pendiente | Pendiente |
| Datos de entrenamiento | NAVER (privado) | Mapillary (open) ✅ |

**Diferencia clave**: SWM-SCL usa Mapillary (datos abiertos) en vez de NAVER Map (privado). Esto hace el pipeline reproducible por cualquiera con un token de Mapillary.

---

## Roadmap

- [x] Demo interactiva con mapa real de Santiago
- [x] Pipeline de scraping Mapillary
- [x] Geo-indexer con BallTree haversine  
- [x] Cross-temporal pairer
- [x] Virtual Lookahead Sink (implementación)
- [x] Configuración completa YAML
- [ ] Fine-tune sobre subset Santiago (en progreso)
- [ ] Datos sintéticos CARLA de calles chilenas
- [ ] Checkpoint público disponible
- [ ] API de inferencia (FastAPI)
- [ ] Integración con Atlas Censal Chile

---

## Aplicaciones Potenciales

- **Planificación urbana**: visualizar propuestas de redesarrollo en calles reales
- **Turismo virtual**: explorar Santiago desde cualquier parte del mundo
- **Gaming / entretenimiento**: niveles generativos basados en Santiago real
- **Autonomous driving**: generar escenarios de prueba en calles chilenas
- **Patrimonio cultural**: preservar zonas históricas en forma dinámica

---

## Créditos

Este proyecto adapta el trabajo de:

> Seo et al., "Grounding World Simulation Models in a Real-World Metropolis" (2025)  
> KAIST AI + NAVER AI Lab  
> [https://seoul-world-model.github.io](https://seoul-world-model.github.io)

El modelo base es [Cosmos-Predict2.5-2B](https://github.com/NVIDIA/Cosmos) de NVIDIA.

Los datos de street-view provienen de [Mapillary](https://www.mapillary.com) bajo sus términos de uso para investigación.

---

## Licencia

MIT License — ver [LICENSE](LICENSE)

Los datos de Mapillary están sujetos a sus propios [términos de uso](https://www.mapillary.com/terms).

---

*Hecho con ☕ y código en Santiago de Chile por [VibeCodingChile](https://vibecodingchile.cl)*  
*Del Código Civil al Claude Code → Del Atlas Censal al Santiago World Model*
