"""
PUNTO DE ENTRADA — FastAPI

Configura la app, el CORS, el logging y los eventos de arranque/apagado.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, ingest, query
from app.api.dependencies import get_vector_store
from app.config import get_settings

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────────────
    logger.info("Iniciando RAG Multimodal API...")

    # Crear carpetas de almacenamiento
    os.makedirs(settings.storage_dir, exist_ok=True)
    os.makedirs(os.path.join(settings.storage_dir, "extracted_images"), exist_ok=True)

    # Inicializar la colección de Qdrant (crea si no existe)
    # El adaptador ya maneja el fallback in-memory si Qdrant no está disponible
    try:
        vs = get_vector_store()
        await vs.initialize()
        mode = getattr(vs, "mode", "unknown")
        logger.info(f"✅ Qdrant inicializado — modo: {mode}")
    except Exception as e:
        logger.error(f"Error al inicializar Qdrant: {e}")

    logger.info("API lista para recibir requests")
    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────
    logger.info("Cerrando RAG Multimodal API...")


app = FastAPI(
    title="RAG Multimodal API",
    description=(
        "Sistema RAG para procesar documentos PDF técnicos con soporte "
        "multimodal (texto, tablas e imágenes). Arquitectura hexagonal con "
        "inyección de dependencias e integración OpenAI/Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — permite que Streamlit (:8501) llame a la API (:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(documents.router)


@app.get("/health", tags=["Sistema"])
async def health_check():
    """
    Healthcheck — usado por Docker para saber si el servicio está listo.
    También muestra en qué modo está corriendo Qdrant.
    """
    vs = get_vector_store()
    qdrant_mode = getattr(vs, "mode", "unknown")
    return {
        "status": "ok",
        "service": "RAG Multimodal API",
        "version": "1.0.0",
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "qdrant_mode": qdrant_mode,   # "server" o "in-memory"
    }
