"""
PUERTOS (Interfaces / Contratos)

En arquitectura hexagonal, los "puertos" son interfaces abstractas
que definen QUÉ debe hacer un componente, sin decir CÓMO lo hace.

Por ejemplo: DocumentParser define que alguien puede parsear un PDF,
pero no dice si usa PyMuPDF, pdfplumber u otra librería.

VENTAJA: Si mañana cambias de Qdrant a ChromaDB, solo cambias el adaptador,
nunca el código de negocio.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from .entities import DocumentChunk, IngestionJob, QueryResult


class DocumentParser(ABC):
    """
    Contrato para parsear documentos PDF.
    Cualquier librería de extracción debe implementar este método.
    """
    @abstractmethod
    async def parse(self, file_path: str, document_id: str, document_name: str) -> list[DocumentChunk]:
        """
        Recibe la ruta al PDF y retorna una lista de chunks extraídos.
        Cada chunk tiene texto, tipo (texto/tabla/imagen) y metadatos de ubicación.
        """
        ...


class VectorStore(ABC):
    """
    Contrato para la base de datos vectorial.
    Puede ser Qdrant, ChromaDB, Weaviate, etc.
    """
    @abstractmethod
    async def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Guarda/actualiza los chunks con sus embeddings en la BD vectorial."""
        ...

    @abstractmethod
    async def hybrid_search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """
        Búsqueda híbrida: combina búsqueda semántica (vectores) con búsqueda
        por palabras clave (BM25). Devuelve los chunks más relevantes.
        """
        ...

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> None:
        """Elimina todos los chunks de un documento por su ID."""
        ...

    @abstractmethod
    async def delete_by_document_name(self, document_name: str) -> int:
        """
        Elimina todos los chunks de un documento por su nombre de archivo.
        Retorna la cantidad de puntos eliminados.
        """
        ...

    @abstractmethod
    async def list_documents(self) -> list[dict]:
        """
        Retorna la lista de documentos indexados con sus estadísticas agregadas
        (total_chunks, total_pages, text_chunks, table_chunks, image_chunks).
        """
        ...


class LLMPort(ABC):
    """
    Contrato para el modelo de lenguaje.
    Puede ser OpenAI GPT, Ollama (local), Anthropic Claude, etc.
    """
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[list[str]] = None,
    ) -> str:
        """
        Llama al LLM con un prompt de sistema, uno de usuario y opcionalmente imágenes/diagramas.
        Devuelve la respuesta como texto sintetizado.
        """
        ...

    @abstractmethod
    async def describe_image(self, image_path: str) -> str:
        """
        Genera una descripción técnica y estructurada de una imagen o diagrama
        para que pueda indexarse vectorialmente con significado semántico.
        """
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Convierte texto en un vector numérico (embedding).
        Este vector captura el SIGNIFICADO semántico del texto.
        """
        ...


class JobRepository(ABC):
    """
    Contrato para guardar y consultar el estado de los trabajos de ingesta.
    En producción podría ser Redis o una BD, aquí usaremos memoria RAM.
    """
    @abstractmethod
    def save(self, job: IngestionJob) -> None:
        """Guarda o actualiza el estado de un job."""
        ...

    @abstractmethod
    def get(self, job_id: str) -> Optional[IngestionJob]:
        """Busca un job por su ID. Retorna None si no existe."""
        ...
