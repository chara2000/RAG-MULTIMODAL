"""
ENDPOINTS DE INGESTA

POST /ingest/        → Sube un PDF, devuelve job_id inmediatamente
GET  /ingest/{id}   → Consulta el estado del job (polling)
"""
import logging
import os
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from app.config import get_settings
from app.api.dependencies import get_ingest_use_case, get_job_repo, get_vector_store
from app.domain.ports import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingesta de Documentos"])


@router.post("/", summary="Subir un PDF para indexar")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Archivo PDF a procesar"),
    overwrite: bool = False,
    ingest_uc=Depends(get_ingest_use_case),
    job_repo=Depends(get_job_repo),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    Recibe un PDF, verifica duplicados, lo guarda en disco y lanza el procesamiento en segundo plano.
    Devuelve el job_id INMEDIATAMENTE sin esperar a que termine el procesamiento.
    """
    settings = get_settings()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    # Verificación de duplicados para evitar re-indexación involuntaria
    try:
        existing_docs = await vector_store.list_documents()
        existing_names = {d["document_name"].lower() for d in existing_docs}
        req_name = file.filename.lower()
        if req_name in existing_names:
            if not overwrite:
                raise HTTPException(
                    status_code=409,
                    detail=f"El documento '{file.filename}' ya se encuentra indexado. Para actualizarlo, active la opción de sobrescribir.",
                )
            else:
                logger.info(f"Sobrescribiendo documento existente: '{file.filename}'")
                await vector_store.delete_by_document_name(file.filename)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"No se pudo verificar duplicados en VectorStore: {e}")

    content = await file.read()
    MAX_SIZE = 50 * 1024 * 1024  # 50 MB
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="El archivo excede el límite de 50MB")

    # Guardar en la carpeta de storage
    storage_path = Path(settings.storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    safe_name = f"{file_id}_{file.filename}"
    file_path = str(storage_path / safe_name)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    logger.info(f"Archivo guardado: {file_path} ({len(content)} bytes)")

    # Crear job y lanzar en background
    job = ingest_uc.create_job(document_name=file.filename)
    background_tasks.add_task(ingest_uc.run, job, file_path)

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "message": f"'{file.filename}' recibido. Procesando en segundo plano.",
    }


@router.get("/{job_id}", summary="Consultar estado de un job")
async def get_job_status(job_id: str, job_repo=Depends(get_job_repo)):
    """
    Consulta el estado actual de un trabajo de ingesta.
    Úsalo en polling desde el frontend hasta que status == 'completed' o 'failed'.
    """
    job = job_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' no encontrado")

    return {
        "job_id": job.job_id,
        "document_name": job.document_name,
        "status": job.status.value,
        "progress_message": job.progress_message,
        "total_chunks": job.total_chunks,
        "error": job.error,
    }
