"""
TEST DE INTEGRACIÓN — API Endpoints

Prueba los endpoints de FastAPI usando el cliente de prueba de httpx.
Usa mocks para los servicios externos (no necesita Qdrant ni OpenAI corriendo).
"""
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

from app.api.main import app
from app.domain.entities import IngestionJob, JobStatus, QueryResult


@pytest.fixture
def mock_ingest_uc():
    """Mock del caso de uso de ingesta."""
    uc = AsyncMock()
    job = IngestionJob(
        job_id="test-job-123",
        document_name="test.pdf",
        status=JobStatus.PENDING,
        progress_message="Esperando procesamiento...",
    )
    uc.create_job = MagicMock(return_value=job)
    uc.run = AsyncMock(return_value=None)
    return uc


@pytest.fixture
def mock_query_uc():
    """Mock del caso de uso de consulta."""
    uc = AsyncMock()
    uc.execute.return_value = QueryResult(
        answer="**Respuesta de prueba**: El sistema funciona correctamente.",
        sources=[{"document_name": "test.pdf", "page_number": 5, "chunk_type": "text"}],
        images=[],
        has_sufficient_context=True,
    )
    return uc


class TestIngestEndpoints:
    """Tests de integración para los endpoints de ingesta."""

    @pytest.mark.asyncio
    async def test_ingest_returns_job_id(self, mock_ingest_uc):
        """PRUEBA: Al subir un PDF, la API devuelve un job_id inmediatamente."""
        from app.api.dependencies import get_ingest_use_case
        app.dependency_overrides[get_ingest_use_case] = lambda: mock_ingest_uc
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Simular subida de un archivo PDF
                files = {"file": ("documento.pdf", b"%PDF-1.4 fake content", "application/pdf")}
                response = await client.post("/ingest/", files=files)

            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["job_id"] == "test-job-123"
        finally:
            app.dependency_overrides.pop(get_ingest_use_case, None)

    @pytest.mark.asyncio
    async def test_ingest_rejects_non_pdf(self):
        """PRUEBA: La API rechaza archivos que no son PDF."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            files = {"file": ("imagen.jpg", b"fake image data", "image/jpeg")}
            response = await client.post("/ingest/", files=files)

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_health_check(self):
        """PRUEBA: El endpoint de salud responde correctamente."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestQueryEndpoints:
    """Tests de integración para los endpoints de consulta."""

    @pytest.mark.asyncio
    async def test_query_returns_answer(self, mock_query_uc):
        """PRUEBA: Una consulta válida devuelve respuesta con fuentes."""
        from app.api.dependencies import get_query_use_case
        app.dependency_overrides[get_query_use_case] = lambda: mock_query_uc
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/query",
                    json={"question": "¿Cómo funciona el motor?", "top_k": 3},
                )

            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert len(data["sources"]) > 0
        finally:
            app.dependency_overrides.pop(get_query_use_case, None)

    @pytest.mark.asyncio
    async def test_query_rejects_empty_question(self):
        """PRUEBA: La API rechaza preguntas vacías."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/query", json={"question": ""})

        assert response.status_code == 400


class TestDocumentEndpoints:
    """Tests de integración para los endpoints de gestión de documentos (CRUD)."""

    @pytest.mark.asyncio
    async def test_list_documents_returns_list(self):
        """PRUEBA: GET /documents/ retorna la lista de documentos."""
        mock_vs = AsyncMock()
        mock_vs.list_documents.return_value = [
            {
                "document_name": "manual.pdf",
                "document_id": "doc-1",
                "total_chunks": 10,
                "total_pages": 5,
                "text_chunks": 8,
                "table_chunks": 1,
                "image_chunks": 1,
            }
        ]
        from app.api.dependencies import get_vector_store
        app.dependency_overrides[get_vector_store] = lambda: mock_vs
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/documents/")

            assert response.status_code == 200
            data = response.json()
            assert "documents" in data
            assert len(data["documents"]) == 1
            assert data["documents"][0]["document_name"] == "manual.pdf"
        finally:
            app.dependency_overrides.pop(get_vector_store, None)

    @pytest.mark.asyncio
    async def test_delete_document_success(self):
        """PRUEBA: DELETE /documents/{name} elimina el documento."""
        mock_vs = AsyncMock()
        mock_vs.delete_by_document_name.return_value = 15
        from app.api.dependencies import get_vector_store
        app.dependency_overrides[get_vector_store] = lambda: mock_vs
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.delete("/documents/manual.pdf")

            assert response.status_code == 200
            data = response.json()
            assert data["deleted"] is True
            assert data["points_deleted"] == 15
        finally:
            app.dependency_overrides.pop(get_vector_store, None)
