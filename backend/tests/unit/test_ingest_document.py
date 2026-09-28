"""
TESTS UNITARIOS — IngestDocumentUseCase

Verifica el flujo de ingesta de documentos: crear job, procesar, actualizar estado.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os
import tempfile

from app.application.ingest_document import IngestDocumentUseCase
from app.domain.entities import ChunkType, DocumentChunk, IngestionJob, JobStatus
from app.infrastructure.jobs.job_repository import InMemoryJobRepository


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="c1",
            document_id="job-123",
            document_name="test.pdf",
            page_number=1,
            chunk_type=ChunkType.TEXT,
            text="Texto de prueba para el test unitario del sistema RAG multimodal.",
        )
    ]


@pytest.fixture
def mock_parser(sample_chunks):
    parser = AsyncMock()
    parser.parse.return_value = sample_chunks
    return parser


@pytest.fixture
def mock_vector_store():
    vs = AsyncMock()
    vs.upsert.return_value = None
    return vs


@pytest.fixture
def job_repo():
    return InMemoryJobRepository()


class TestIngestDocumentUseCase:
    """Tests del caso de uso de ingesta de documentos."""

    def test_create_job_returns_pending_status(self, mock_parser, mock_vector_store, job_repo, tmp_path):
        """PRUEBA: Al crear un job, su estado inicial es PENDING."""
        use_case = IngestDocumentUseCase(
            parser=mock_parser,
            vector_store=mock_vector_store,
            job_repo=job_repo,
            upload_dir=str(tmp_path),
        )

        job = use_case.create_job("documento_tecnico.pdf")

        assert job.status == JobStatus.PENDING
        assert job.document_name == "documento_tecnico.pdf"
        assert job.job_id is not None

    def test_create_job_is_persisted(self, mock_parser, mock_vector_store, job_repo, tmp_path):
        """PRUEBA: El job creado se puede consultar por su ID."""
        use_case = IngestDocumentUseCase(
            parser=mock_parser,
            vector_store=mock_vector_store,
            job_repo=job_repo,
            upload_dir=str(tmp_path),
        )

        job = use_case.create_job("test.pdf")
        retrieved = job_repo.get(job.job_id)

        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    @pytest.mark.asyncio
    async def test_successful_ingest_sets_completed(
        self, mock_parser, mock_vector_store, job_repo, tmp_path
    ):
        """PRUEBA: Una ingesta exitosa cambia el estado a COMPLETED."""
        use_case = IngestDocumentUseCase(
            parser=mock_parser,
            vector_store=mock_vector_store,
            job_repo=job_repo,
            upload_dir=str(tmp_path),
        )

        # Crear un archivo PDF falso para el test
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"fake pdf content")

        job = use_case.create_job("test.pdf")
        await use_case.run(job, str(fake_pdf))

        updated_job = job_repo.get(job.job_id)
        assert updated_job.status == JobStatus.COMPLETED
        assert updated_job.total_chunks == 1

    @pytest.mark.asyncio
    async def test_parser_error_sets_failed_status(
        self, mock_vector_store, job_repo, tmp_path
    ):
        """PRUEBA: Si el parser falla, el job queda en estado FAILED con el error."""
        failing_parser = AsyncMock()
        failing_parser.parse.side_effect = Exception("PDF corrupto o inaccesible")

        use_case = IngestDocumentUseCase(
            parser=failing_parser,
            vector_store=mock_vector_store,
            job_repo=job_repo,
            upload_dir=str(tmp_path),
        )

        fake_pdf = tmp_path / "corrupted.pdf"
        fake_pdf.write_bytes(b"not a real pdf")

        job = use_case.create_job("corrupted.pdf")
        await use_case.run(job, str(fake_pdf))

        updated_job = job_repo.get(job.job_id)
        assert updated_job.status == JobStatus.FAILED
        assert updated_job.error is not None
        assert "PDF corrupto" in updated_job.error

    @pytest.mark.asyncio
    async def test_upsert_called_after_parse(
        self, mock_parser, mock_vector_store, job_repo, tmp_path, sample_chunks
    ):
        """PRUEBA: El VectorStore recibe los chunks que generó el parser."""
        use_case = IngestDocumentUseCase(
            parser=mock_parser,
            vector_store=mock_vector_store,
            job_repo=job_repo,
            upload_dir=str(tmp_path),
        )

        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"fake pdf")

        job = use_case.create_job("test.pdf")
        await use_case.run(job, str(fake_pdf))

        # Verificar que upsert fue llamado con los chunks correctos
        mock_vector_store.upsert.assert_called_once_with(sample_chunks)
