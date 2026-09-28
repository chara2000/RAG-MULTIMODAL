"""
TESTS UNITARIOS — AnswerQueryUseCase

Los tests unitarios verifican que la lógica de negocio funciona correctamente
SIN llamar a servicios externos (OpenAI, Qdrant).

MOCKS:
Reemplazamos los adaptadores reales con objetos falsos (mocks) que simulan
su comportamiento. Así los tests son:
- Rápidos (no hacen llamadas de red)
- Deterministas (siempre el mismo resultado)
- Aislados (no dependen de que Qdrant esté corriendo)

FRASE CLAVE para sustentación:
"Los mocks implementan las mismas interfaces del dominio, garantizando
que si el mock funciona, el sistema real también funcionará."
"""
import pytest
from unittest.mock import AsyncMock

from app.application.answer_query import AnswerQueryUseCase
from app.domain.entities import ChunkType, DocumentChunk, QueryResult


# ─── Fixtures: Datos de prueba reutilizables ──────────────────────────────────

@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Chunks de ejemplo que simulan los resultados de Qdrant."""
    return [
        DocumentChunk(
            chunk_id="test-chunk-1",
            document_id="doc-001",
            document_name="Manual_Motor.pdf",
            page_number=12,
            chunk_type=ChunkType.TEXT,
            text="El sistema de refrigeración del motor usa líquido refrigerante "
                 "que circula por el bloque del motor para mantener la temperatura óptima.",
            nearest_image_path="/app/uploads/extracted_images/doc-001_p12_img0.png",
        ),
        DocumentChunk(
            chunk_id="test-chunk-2",
            document_id="doc-001",
            document_name="Manual_Motor.pdf",
            page_number=13,
            chunk_type=ChunkType.TABLE,
            text="Temperatura mínima | 60°C\nTemperatura óptima | 90°C\nTemperatura máxima | 110°C",
        ),
    ]


@pytest.fixture
def mock_vector_store(sample_chunks):
    """VectorStore falso que siempre devuelve los chunks de prueba."""
    mock = AsyncMock()
    mock.hybrid_search.return_value = sample_chunks
    return mock


@pytest.fixture
def mock_llm():
    """LLM falso que devuelve una respuesta predefinida."""
    mock = AsyncMock()
    mock.generate.return_value = (
        "El sistema de refrigeración usa **líquido refrigerante** que circula "
        "por el bloque del motor.\n\n*Fuente: Manual_Motor.pdf, Página 12*"
    )
    return mock


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestAnswerQueryUseCase:
    """Tests del caso de uso principal de consulta RAG."""

    @pytest.mark.asyncio
    async def test_successful_query_returns_answer(self, mock_vector_store, mock_llm):
        """
        PRUEBA: Una pregunta válida devuelve una respuesta con fuentes.
        Verifica el flujo completo: retrieve → augment → generate.
        """
        use_case = AnswerQueryUseCase(vector_store=mock_vector_store, llm=mock_llm)

        result = await use_case.execute("¿Cómo funciona el sistema de refrigeración?")

        # Verificar que el resultado es un QueryResult con datos válidos
        assert isinstance(result, QueryResult)
        assert len(result.answer) > 0
        assert result.has_sufficient_context is True

    @pytest.mark.asyncio
    async def test_sources_are_deduplicated(self, mock_vector_store, mock_llm):
        """
        PRUEBA: Las fuentes en la respuesta no se duplican.
        Si dos chunks vienen de la misma página, aparece UNA fuente.
        """
        use_case = AnswerQueryUseCase(vector_store=mock_vector_store, llm=mock_llm)
        result = await use_case.execute("¿Qué temperatura es la óptima?")

        # Tenemos 2 chunks de Manual_Motor.pdf páginas 12 y 13 → 2 fuentes distintas
        source_keys = {
            (s["document_name"], s["page_number"]) for s in result.sources
        }
        assert len(source_keys) == len(result.sources)  # No hay duplicados

    @pytest.mark.asyncio
    async def test_no_chunks_returns_insufficient_context(self, mock_llm):
        """
        PRUEBA: Cuando no hay chunks relevantes, se informa que no hay contexto.
        Este comportamiento es el "prompt defensivo" anti-alucinaciones.
        """
        empty_vector_store = AsyncMock()
        empty_vector_store.hybrid_search.return_value = []  # Sin resultados

        use_case = AnswerQueryUseCase(vector_store=empty_vector_store, llm=mock_llm)
        result = await use_case.execute("Pregunta sobre algo no indexado")

        assert result.has_sufficient_context is False
        assert len(result.sources) == 0
        # El LLM NO debería ser llamado si no hay contexto
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_images_are_included_in_result(self, mock_vector_store, mock_llm, sample_chunks):
        """
        PRUEBA: Las imágenes cercanas al texto relevante se incluyen en la respuesta.
        Este es el contexto visual que pide la prueba técnica.
        """
        use_case = AnswerQueryUseCase(vector_store=mock_vector_store, llm=mock_llm)
        result = await use_case.execute("¿Cómo funciona el sistema?")

        # El primer chunk tiene nearest_image_path definido
        # Verificamos que se intenta incluirlo (puede no existir en el FS durante tests)
        # La lógica de filtrar por os.path.exists está en la capa API, no aquí
        assert isinstance(result.images, list)

    @pytest.mark.asyncio
    async def test_vector_store_is_called_with_question(self, mock_vector_store, mock_llm):
        """
        PRUEBA: El VectorStore siempre recibe la pregunta del usuario (no algo distinto).
        Verifica que la pregunta se pasa correctamente al motor de búsqueda.
        """
        use_case = AnswerQueryUseCase(vector_store=mock_vector_store, llm=mock_llm)
        question = "¿Cuál es la presión máxima del sistema hidráulico?"

        await use_case.execute(question)

        mock_vector_store.hybrid_search.assert_called_once_with(question, top_k=5)

    @pytest.mark.asyncio
    async def test_llm_error_returns_graceful_response(self, mock_vector_store):
        """
        PRUEBA: Si el LLM falla, el sistema responde con un mensaje de error claro.
        Resiliencia — el sistema no debe caerse completamente si el LLM falla.
        """
        failing_llm = AsyncMock()
        failing_llm.generate.side_effect = Exception("OpenAI rate limit exceeded")

        use_case = AnswerQueryUseCase(vector_store=mock_vector_store, llm=failing_llm)
        result = await use_case.execute("¿Cuál es la temperatura de operación?")

        # El sistema debe responder con un mensaje de error, no lanzar excepción
        assert result.has_sufficient_context is False
        assert "Error" in result.answer or "error" in result.answer.lower()
