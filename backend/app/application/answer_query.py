"""
CASO DE USO: ANSWER QUERY

Este es el flujo RAG principal. Cuando el usuario hace una pregunta:

1. RETRIEVE: Busca en Qdrant los chunks más relevantes (búsqueda híbrida)
2. AUGMENT: Construye un prompt con esos chunks como contexto
3. GENERATE: Llama al LLM para que genere la respuesta

PROMPT DEFENSIVO:
El prompt le ordena explícitamente al LLM:
- Solo usar la información provista en el contexto
- Decir "No tengo información suficiente" si los chunks no responden la pregunta
- Nunca inventar datos que no estén en el contexto

Esto mitiga las "alucinaciones" — el mayor problema de los LLMs.
"""
import logging
import os
import re
from typing import Optional

from app.domain.entities import ChunkType, QueryResult
from app.domain.ports import LLMPort, VectorStore

logger = logging.getLogger(__name__)

# Prompt del sistema — guía el comportamiento del LLM (Multimodal)
SYSTEM_PROMPT = """Eres un asistente técnico experto en análisis de documentos y manuales de ingeniería.
Tu función es responder preguntas basándote rigurosamente en los fragmentos de texto, tablas e imágenes provistos.

REGLAS ESTRICTAS DE RESPUESTA Y CITAS:
1. Basa tu explicación exclusivamente en la información técnica del contexto y en las imágenes/diagramas adjuntos.
2. Si el usuario solicita detalles de componentes, diagramas, esquemas o controles (ej: velocímetro, interruptores, piezas):
   - Describe de manera técnica y estructurada los componentes, indicadores, testigos o funciones visibles en las imágenes o en el texto.
   - Explica la función de cada elemento señalado.
3. CITAS Y NÚMEROS DE PÁGINA (CRÍTICO):
   - Cita SIEMPRE el nombre del documento y el número de página EXACTO que figura en el encabezado del fragmento provisto (ej: si el encabezado dice '[Fragmento 1] Fuente: manual.pdf, Página 14', tu cita DEBE indicar 'Página 14').
   - NUNCA inventes números de página ni uses números distintos a los indicados en los metadatos de los fragmentos provistos.
4. Menciona las imágenes o diagramas complementarios que se adjuntan a la respuesta indicando su página correspondiente.
5. Solo si el contexto y las imágenes no tienen relación alguna con la pregunta, responde: "No encontré información suficiente en los documentos disponibles para responder esto."
6. Responde en el mismo idioma que la consulta usando formato Markdown estructurado (viñetas, negritas, subtítulos claros).
"""


class AnswerQueryUseCase:
    """
    Caso de uso para responder preguntas usando RAG.

    PRINCIPIO: La lógica de negocio no sabe si usamos OpenAI o Ollama,
    ni si el vector store es Qdrant o ChromaDB. Solo habla con interfaces.
    """

    def __init__(self, vector_store: VectorStore, llm: LLMPort):
        self.vector_store = vector_store
        self.llm = llm

    async def execute(
        self,
        question: str,
        top_k: int = 5,
        llm_override: Optional[LLMPort] = None,
    ) -> QueryResult:
        """
        Ejecuta el pipeline RAG completo para responder una pregunta.

        Args:
            question: La pregunta del usuario en lenguaje natural
            top_k: Número de chunks a recuperar para el contexto
            llm_override: Adaptador LLM opcional para cambiar dinámicamente de proveedor

        Returns:
            QueryResult con la respuesta, fuentes e imágenes relevantes
        """
        active_llm = llm_override or self.llm
        logger.info(f"Procesando consulta con LLM [{type(active_llm).__name__}]: '{question[:80]}...'")

        # PASO 1 — RETRIEVE: Buscar chunks relevantes
        raw_chunks = await self.vector_store.hybrid_search(question, top_k=top_k)

        if not raw_chunks:
            logger.warning("No se encontraron chunks relevantes para la consulta")
            return QueryResult(
                answer="No encontré información relevante en los documentos indexados. "
                       "Asegúrate de haber subido documentos primero.",
                sources=[],
                images=[],
                has_sufficient_context=False,
            )

        # PASO 1.5 — DEDUPLICACIÓN DE CHUNKS RETRIEVE
        # Previene chunks idénticos si un mismo documento fue indexado con diferentes IDs
        chunks = []
        seen_chunk_signatures = set()
        for c in raw_chunks:
            clean_name = re.sub(r"^[0-9a-fA-F-]{36}_", "", c.document_name)
            sig = (clean_name.lower(), c.page_number, c.chunk_type.value, c.text[:80])
            if sig not in seen_chunk_signatures:
                seen_chunk_signatures.add(sig)
                c.document_name = clean_name
                chunks.append(c)

        # PASO 2 — AUGMENT: Construir el contexto para el LLM
        context_parts = []
        sources = []
        images = []
        seen_sources = set()
        seen_image_signatures = set()

        for i, chunk in enumerate(chunks, 1):
            clean_doc = re.sub(r"^[0-9a-fA-F-]{36}_", "", chunk.document_name)

            # Formatear el contexto con metadatos claros
            context_parts.append(
                f"[Fragmento {i}] "
                f"Fuente: {clean_doc}, Página {chunk.page_number}, "
                f"Tipo: {chunk.chunk_type.value}\n"
                f"{chunk.text}"
            )

            # Recopilar fuentes únicas (documento + página)
            source_key = f"{clean_doc.lower()}:{chunk.page_number}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "document_name": clean_doc,
                    "page_number": chunk.page_number,
                    "chunk_type": chunk.chunk_type.value,
                })

            # Recopilar imágenes relevantes sin repeticiones de la misma página
            img_path = chunk.image_path or chunk.nearest_image_path
            if img_path:
                filename = os.path.basename(img_path)
                sig_match = re.search(r"(_p\d+_img\d+)", filename)
                img_sig = sig_match.group(1) if sig_match else filename

                if img_sig not in seen_image_signatures and img_path not in images:
                    seen_image_signatures.add(img_sig)
                    images.append(img_path)

        context = "\n\n---\n\n".join(context_parts)

        user_prompt = f"""Contexto de los documentos:

{context}

---

Pregunta del usuario: {question}

Responde basándote únicamente en el contexto anterior. Recuerda citar fielmente los números de página exactos provistos en los fragmentos."""

        # PASO 3 — GENERATE: Llamar al LLM con el contexto y las imágenes
        try:
            answer = await active_llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                images=images[:6],
            )
            negative_phrases = [
                "no encontré información suficiente",
                "no tengo información suficiente",
                "no se encontró información",
                "no encontré información relevante",
                "no hay información suficiente",
                "no contiene información",
                "no se menciona",
                "no se encuentra información",
            ]
            has_context = not any(phrase in answer.lower() for phrase in negative_phrases)
        except Exception as e:
            logger.error(f"Error llamando al LLM: {e}")
            answer = f"Error al generar respuesta: {str(e)}. Por favor intenta nuevamente."
            has_context = False

        # Si el LLM determinó que NO hay contexto suficiente, NO mostrar imágenes ni fuentes irrelevantes
        final_images = images[:6] if has_context else []
        final_sources = sources if has_context else []

        logger.info(
            f"Consulta respondida. Fuentes: {len(final_sources)}, Imágenes: {len(final_images)}, HasContext: {has_context}"
        )

        return QueryResult(
            answer=answer,
            sources=final_sources,
            images=final_images,
            has_sufficient_context=has_context,
        )
