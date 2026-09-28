"""
INYECCIÓN DE DEPENDENCIAS

Decide QUÉ implementación concreta usar según la configuración.
- LLM_PROVIDER=openai  → OpenAIAdapter
- LLM_PROVIDER=ollama  → OllamaAdapter
- QDRANT_HOST=qdrant   → conecta al servidor Docker
- QDRANT_HOST=localhost (sin Docker) → modo in-memory automático
"""
import logging
from functools import lru_cache

from app.application.answer_query import AnswerQueryUseCase
from app.application.ingest_document import IngestDocumentUseCase
from app.config import get_settings
from app.domain.ports import JobRepository, LLMPort, VectorStore
from app.infrastructure.jobs.job_repository import InMemoryJobRepository
from app.infrastructure.llm.ollama_adapter import OllamaAdapter
from app.infrastructure.llm.openai_adapter import OpenAIAdapter
from app.infrastructure.parsing.pymupdf_parser import PyMuPDFParser
from app.infrastructure.vectorstore.qdrant_adapter import QdrantAdapter

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_job_repo() -> JobRepository:
    return InMemoryJobRepository()


@lru_cache(maxsize=1)
def get_openai_adapter() -> OpenAIAdapter:
    settings = get_settings()
    return OpenAIAdapter(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
    )


@lru_cache(maxsize=1)
def get_ollama_adapter() -> OllamaAdapter:
    settings = get_settings()
    return OllamaAdapter(base_url=settings.ollama_base_url, model=settings.ollama_model)


def get_llm(provider: str = None) -> LLMPort:
    """
    Fábrica de LLM — permite intercambio fluido de componentes (DIP).
    Soporta 'openai' y 'ollama'. Si no se especifica, usa LLM_PROVIDER de .env.
    """
    settings = get_settings()
    target = (provider or settings.llm_provider).lower().strip()
    if target == "ollama":
        return get_ollama_adapter()
    else:
        return get_openai_adapter()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """
    Fábrica del VectorStore.
    Si Qdrant no está disponible, QdrantAdapter activa in-memory automáticamente.
    """
    settings = get_settings()
    llm = get_llm()
    return QdrantAdapter(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
        llm=llm,
    )


@lru_cache(maxsize=1)
def get_parser() -> PyMuPDFParser:
    settings = get_settings()
    return PyMuPDFParser(storage_dir=settings.storage_dir)


def get_ingest_use_case() -> IngestDocumentUseCase:
    settings = get_settings()
    return IngestDocumentUseCase(
        parser=get_parser(),
        vector_store=get_vector_store(),
        job_repo=get_job_repo(),
        storage_dir=settings.storage_dir,
        llm=get_llm(),
    )


def get_query_use_case() -> AnswerQueryUseCase:
    return AnswerQueryUseCase(
        vector_store=get_vector_store(),
        llm=get_llm(),
    )
