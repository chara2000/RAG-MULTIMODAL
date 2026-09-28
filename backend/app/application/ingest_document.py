"""
CASO DE USO: INGEST DOCUMENT

Este es el "cerebro" del procesamiento de documentos.
Orquesta: Parser → VectorStore → JobRepository

ASINCRONÍA:
La función run() se ejecuta en SEGUNDO PLANO usando asyncio.
Cuando el usuario sube un PDF, la API responde INMEDIATAMENTE con un job_id.
El procesamiento real sucede en paralelo sin bloquear a otros usuarios.

FLUJO:
1. API crea el job (estado: PENDING) y lo guarda
2. API devuelve job_id al cliente
3. En background: job cambia a PROCESSING
4. Parser extrae chunks del PDF
5. VectorStore guarda los chunks con embeddings
6. job cambia a COMPLETED (o FAILED si hay error)
"""
import asyncio
import logging
import os
import uuid
from pathlib import Path

from typing import Optional
from app.domain.entities import ChunkType, IngestionJob, JobStatus
from app.domain.ports import DocumentParser, JobRepository, LLMPort, VectorStore

logger = logging.getLogger(__name__)


class IngestDocumentUseCase:
    """
    Caso de uso para ingestar (procesar e indexar) un documento PDF.

    Principio de Responsabilidad Única (SRP):
    Solo se encarga de coordinar el flujo de ingesta.
    No sabe cómo parsear PDFs (eso es el Parser),
    ni cómo guardar vectores (eso es el VectorStore).
    """

    def __init__(
        self,
        parser: DocumentParser,
        vector_store: VectorStore,
        job_repo: JobRepository,
        storage_dir: str = "",
        upload_dir: str = "",
        llm: Optional[LLMPort] = None,
    ):
        self.parser = parser
        self.vector_store = vector_store
        self.job_repo = job_repo
        self.storage_dir = storage_dir or upload_dir or "./storage"
        self.llm = llm

    def create_job(self, document_name: str) -> IngestionJob:
        """
        Crea un nuevo job de ingesta y lo registra.
        job_id se genera automáticamente en la entidad IngestionJob.
        """
        job = IngestionJob(
            document_name=document_name,
            status=JobStatus.PENDING,
        )
        self.job_repo.save(job)
        logger.info(f"Job creado: {job.job_id} para '{document_name}'")
        return job

    async def run(self, job: IngestionJob, file_path: str) -> None:
        """
        Ejecuta el proceso de ingesta completo en segundo plano.
        Actualiza el estado del job en cada etapa.
        """
        # Actualizar estado a PROCESSING
        job.status = JobStatus.PROCESSING
        job.progress_message = "Extrayendo contenido del PDF..."
        self.job_repo.save(job)

        try:
            # PASO 1: Parsear el PDF → obtener chunks
            logger.info(f"[Job {job.job_id}] Iniciando parseo de {file_path}")
            chunks = await self.parser.parse(
                file_path=file_path,
                document_id=job.job_id,
                document_name=job.document_name,
            )

            if not chunks:
                raise ValueError("El PDF no produjo ningún chunk. Verifica que tenga texto.")

            logger.info(f"[Job {job.job_id}] {len(chunks)} chunks extraídos")

            # PASO 1.5: Enriquecimiento Multimodal de Imágenes
            # Si hay chunks de tipo IMAGE sin texto descriptivo,
            # usamos el modelo de visión para extraer descripciones técnicas
            image_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMAGE and c.image_path]
            if image_chunks and self.llm:
                total_imgs = len(image_chunks)
                job.progress_message = f"Analizando {total_imgs} diagramas/imágenes con visión..."
                self.job_repo.save(job)
                logger.info(f"[Job {job.job_id}] Enriqueciendo {total_imgs} imágenes con visión...")

                for idx, img_chunk in enumerate(image_chunks):
                    # Si el texto es genérico (ej. solo dice [Imagen en página X...])
                    if "[Imagen" in img_chunk.text and len(img_chunk.text.strip()) < 60:
                        try:
                            desc = await self.llm.describe_image(img_chunk.image_path)
                            if desc:
                                img_chunk.text = (
                                    f"[Diagrama/Imagen Técnico - Página {img_chunk.page_number}]: {desc}"
                                )
                                logger.info(
                                    f"[Job {job.job_id}] Img pág {img_chunk.page_number} descrita: {desc[:60]}..."
                                )
                            # Pausa corta para prevenir rate limits de tokens por minuto (TPM)
                            await asyncio.sleep(0.3)
                        except Exception as e:
                            logger.warning(f"Error describiendo imagen {img_chunk.image_path}: {e}")

            # Actualizar progreso
            job.progress_message = f"Indexando {len(chunks)} fragmentos en la base vectorial..."
            job.total_chunks = len(chunks)
            self.job_repo.save(job)

            # PASO 2: Guardar en la base de datos vectorial
            await self.vector_store.upsert(chunks)

            # PASO 3: Marcar como completado
            job.status = JobStatus.COMPLETED
            job.progress_message = f"Completado. {len(chunks)} fragmentos indexados."
            self.job_repo.save(job)
            logger.info(f"[Job {job.job_id}] Ingesta completada exitosamente")

        except Exception as e:
            # Si algo falla, registramos el error y marcamos el job como FAILED
            error_msg = str(e)
            logger.error(f"[Job {job.job_id}] Error en ingesta: {error_msg}", exc_info=True)
            job.status = JobStatus.FAILED
            job.error = error_msg
            job.progress_message = f"Error: {error_msg}"
            self.job_repo.save(job)

        finally:
            # Limpiar el archivo temporal del disco
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
