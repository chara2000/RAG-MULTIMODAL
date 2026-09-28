"""
ENDPOINTS DE CONSULTA RAG y SERVICIO DE IMÁGENES

POST /query            → Pregunta al sistema RAG
GET  /images/{nombre}  → Sirve imágenes extraídas (el frontend las pide aquí)
"""
import logging
import os
from pathlib import Path

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.dependencies import get_llm, get_query_use_case
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Consultas RAG"])


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    provider: Optional[str] = None  # "openai" o "ollama"

    class Config:
        json_schema_extra = {
            "example": {
                "question": "¿Cómo funciona el sistema de refrigeración?",
                "top_k": 5,
                "provider": "openai",
            }
        }


@router.post("/query", summary="Hacer una pregunta al sistema RAG")
async def query_documents(
    request: QueryRequest,
    query_uc=Depends(get_query_use_case),
):
    """
    Pipeline RAG completo:
    1. Busca chunks relevantes en Qdrant (búsqueda híbrida)
    2. Construye el contexto para el LLM con citas verificadas
    3. Permite elegir dinámicamente el LLM (OpenAI vs Ollama)
    4. Genera respuesta con prompt defensivo anti-alucinaciones
    5. Retorna respuesta + fuentes deduplicadas + URLs de imágenes únicas
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")

    if len(request.question) > 2000:
        raise HTTPException(status_code=400, detail="La pregunta excede 2000 caracteres")

    llm_instance = get_llm(request.provider) if request.provider else None

    result = await query_uc.execute(
        question=request.question,
        top_k=min(request.top_k, 10),
        llm_override=llm_instance,
    )

    # Deduplicar URLs de imágenes preservando orden de relevancia
    image_urls = []
    seen_img_names = set()
    for img in result.images:
        if img and os.path.exists(img):
            fname = Path(img).name
            if fname not in seen_img_names:
                seen_img_names.add(fname)
                image_urls.append(f"/images/{fname}")

    return {
        "answer": result.answer,
        "sources": result.sources,
        "has_sufficient_context": result.has_sufficient_context,
        "image_urls": image_urls,
    }


@router.get("/images/{filename}", summary="Obtener imagen extraída de un PDF")
async def get_image(filename: str):
    """
    Sirve las imágenes extraídas de los PDFs.
    El frontend las pide aquí para mostrarlas junto a las respuestas.

    SEGURIDAD: os.path.basename previene path traversal (../../etc/passwd).
    """
    settings = get_settings()
    safe_filename = os.path.basename(filename)
    image_path = Path(settings.storage_dir) / "extracted_images" / safe_filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Imagen '{safe_filename}' no encontrada")

    return FileResponse(path=str(image_path), media_type="image/png")
