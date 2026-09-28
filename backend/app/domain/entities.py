"""
CAPA DE DOMINIO — Entidades del negocio

Aquí definimos los conceptos puros del sistema RAG:
- DocumentChunk: Un fragmento de texto/imagen extraído del PDF
- IngestionJob: El trabajo de procesamiento de un documento
- QueryResult: La respuesta del sistema a una pregunta

IMPORTANTE: Este módulo NO importa nada de FastAPI, OpenAI ni Qdrant.
Es código Python puro. Así si cambias de framework, el negocio no cambia.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


class JobStatus(str, Enum):
    """Estados posibles de un trabajo de ingesta."""
    PENDING = "pending"       # Recién creado, esperando procesarse
    PROCESSING = "processing" # Se está extrayendo el PDF
    COMPLETED = "completed"   # Listo y almacenado en la BD vectorial
    FAILED = "failed"         # Algo salió mal


class ChunkType(str, Enum):
    """Tipo de contenido de un fragmento del documento."""
    TEXT = "text"    # Párrafo o sección de texto
    TABLE = "table"  # Una tabla extraída del PDF
    IMAGE = "image"  # Una imagen o diagrama


@dataclass
class BoundingBox:
    """
    Coordenadas espaciales de un elemento dentro de la página del PDF.
    Usamos esto para saber DÓNDE estaba el texto o imagen en la página.
    x0, y0 = esquina superior izquierda
    x1, y1 = esquina inferior derecha
    """
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class DocumentChunk:
    """
    Un fragmento del documento listo para ser indexado.

    Contiene el texto, metadatos de dónde vino (página, posición)
    y opcionalmente la ruta a la imagen más cercana (para contexto visual).
    """
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    document_name: str = ""
    page_number: int = 0
    chunk_type: ChunkType = ChunkType.TEXT
    text: str = ""
    image_path: Optional[str] = None      # Ruta local de la imagen si es tipo IMAGE
    bounding_box: Optional[BoundingBox] = None
    nearest_image_path: Optional[str] = None  # Imagen más cercana en la misma página


@dataclass
class IngestionJob:
    """
    Representa el proceso de ingesta de un documento PDF.
    Guarda el estado para que el cliente pueda consultar el progreso.
    """
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_name: str = ""
    status: JobStatus = JobStatus.PENDING
    progress_message: str = "Esperando procesamiento..."
    error: Optional[str] = None
    total_chunks: int = 0


@dataclass
class QueryResult:
    """
    La respuesta del sistema RAG a una pregunta del usuario.
    Incluye la respuesta, de dónde viene la información y si hay imágenes.
    """
    answer: str
    sources: list[dict]          # Lista de {document_name, page_number, chunk_type}
    images: list[str]            # Rutas a imágenes relevantes encontradas
    has_sufficient_context: bool # True si encontró información, False si no
