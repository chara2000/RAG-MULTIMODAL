"""
ADAPTADOR QDRANT — Con Modo In-Memory (Fallback Inteligente)

═══════════════════════════════════════════════════════
¿POR QUÉ TIENE DOS MODOS?
═══════════════════════════════════════════════════════

MODO 1 — Con Docker corriendo:
  El constructor intenta conectar a Qdrant en el host configurado.
  Si responde → usa ese servidor → los datos persisten en disco.

MODO 2 — Sin Docker (desarrollo local / tests):
  Si la conexión falla → activa Qdrant IN-MEMORY.
  La librería qdrant-client incluye un motor en Python puro
  que vive dentro de la RAM. No necesitas instalar nada extra.
  Limitación: los datos se borran cuando cierras el proceso Python.

PARA LA SUSTENTACIÓN:
  "Implementamos Graceful Degradation (Degradación Elegante).
   El sistema detecta si Qdrant está disponible como servidor.
   Si no lo está, conmuta automáticamente a modo in-memory,
   permitiendo que los tests unitarios corran en CI/CD sin
   necesitar Docker, y que los desarrolladores prueben sin
   levantar todos los servicios."
"""
import logging
import os
import re
import uuid
from typing import Optional

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.domain.entities import BoundingBox, ChunkType, DocumentChunk
from app.domain.ports import LLMPort, VectorStore

logger = logging.getLogger(__name__)

STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "a", "al",
    "con", "por", "para", "sobre", "entre", "sin", "tras", "durante", "hasta", "hacia",
    "desde", "según", "mediante", "contra", "bajo", "ante", "que", "qué", "quien", "quién",
    "cual", "cuál", "como", "cómo", "donde", "dónde", "cuando", "cuándo", "y", "o", "u", "e",
    "pero", "mas", "más", "si", "sí", "no", "es", "son", "fue", "era", "ser", "estar",
    "haber", "hay", "tiene", "tienen", "muestra", "muestrame", "muéstrame", "dime", "cuenta",
    "todo", "toda", "todos", "todas", "información", "informacion"
}

VECTOR_SIZE = 1536  # Dimensión de embeddings de text-embedding-3-small


class QdrantAdapter(VectorStore):
    """
    Adaptador para Qdrant.
    Soporta dos modos: servidor Docker y fallback in-memory.
    """

    def __init__(self, host: str, port: int, collection_name: str, llm: LLMPort):
        self.collection_name = collection_name
        self.llm = llm
        self._in_memory = False

        # ─── INTENTAR CONECTAR AL SERVIDOR QDRANT ───────────────────
        try:
            # Usamos el cliente SÍNCRONO solo para el test de conectividad
            # (el cliente async no tiene ping fácil antes del primer await)
            test_client = QdrantClient(host=host, port=port, timeout=2.0)
            test_client.get_collections()   # Lanza excepción si no conecta
            test_client.close()

            # Conexión exitosa → usar servidor real
            self.client = AsyncQdrantClient(host=host, port=port)
            logger.info(f"✅ Qdrant conectado en {host}:{port} (modo servidor)")

        except Exception as e:
            # ─── FALLBACK: MODO IN-MEMORY ────────────────────────────
            logger.warning(
                f"⚠️  Qdrant no disponible en {host}:{port} ({type(e).__name__}). "
                f"Activando modo IN-MEMORY (datos temporales en RAM)."
            )
            self.client = AsyncQdrantClient(location=":memory:")
            self._in_memory = True
            logger.info("✅ Qdrant IN-MEMORY activo. Los datos se perderán al cerrar el proceso.")

    @property
    def mode(self) -> str:
        """Retorna el modo actual para mostrarlo en el health endpoint."""
        return "in-memory" if self._in_memory else "server"

    async def initialize(self) -> None:
        """Crea la colección en Qdrant si no existe."""
        try:
            await self.client.get_collection(self.collection_name)
            logger.info(f"Colección '{self.collection_name}' ya existe")
        except Exception:
            logger.info(f"Creando colección '{self.collection_name}'...")
            try:
                test_emb = await self.llm.embed("dimension test")
                dim = len(test_emb)
            except Exception:
                dim = VECTOR_SIZE
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Colección creada exitosamente con dimensión {dim}")

    async def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Guarda chunks con sus embeddings en Qdrant."""
        if not chunks:
            return

        points = []
        for chunk in chunks:
            try:
                vector = await self.llm.embed(chunk.text[:8000])
            except Exception as e:
                logger.warning(f"Error generando embedding para chunk {chunk.chunk_id}: {e}")
                continue

            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_type": chunk.chunk_type.value,
                "text": chunk.text,
                "image_path": chunk.image_path,
                "nearest_image_path": chunk.nearest_image_path,
            }
            if chunk.bounding_box:
                payload.update({
                    "bbox_x0": chunk.bounding_box.x0,
                    "bbox_y0": chunk.bounding_box.y0,
                    "bbox_x1": chunk.bounding_box.x1,
                    "bbox_y1": chunk.bounding_box.y1,
                })

            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            ))

        if points:
            await self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Guardados {len(points)} chunks en Qdrant [{self.mode}]")

    async def hybrid_search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """Búsqueda híbrida avanzada: filtrado de páginas + semántica + palabras clave."""
        seen_ids: set = set()
        combined: list[DocumentChunk] = []

        # ── 1. DETECCIÓN Y FILTRADO INTELIGENTE POR NÚMERO DE PÁGINA ───────────
        # Ejemplos: "página 11", "pagina 141", "pag 15", "pág 8", "hoja 12"
        page_matches = re.findall(r'(?:p[aá]g(?:ina)?|page|hoja)\s*[:.]?\s*(\d+)', query, re.IGNORECASE)
        if page_matches and not self._in_memory:
            base_page = int(page_matches[0])
            # Consideramos la página exacta y un margen (+/- 2 páginas)
            # para compensar el desfase entre el número impreso del libro (folio) y la hoja física del PDF
            target_pages = [
                base_page,
                base_page + 1,
                base_page + 2,
                max(1, base_page - 1),
                max(1, base_page - 2),
            ]
            try:
                page_scroll, _ = await self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        should=[
                            models.FieldCondition(
                                key="page_number",
                                match=models.MatchValue(value=p),
                            )
                            for p in target_pages
                        ]
                    ),
                    limit=top_k * 3,
                    with_payload=True,
                )

                q_lower = query.lower()

                def page_priority(item):
                    p_num = item.payload.get("page_number", 0)
                    doc_name = item.payload.get("document_name", "").lower()
                    doc_boost = (
                        10
                        if any(term in doc_name for term in ["manual", "bajaj", "pulsar"] if term in q_lower)
                        or any(term in doc_name for term in ["perro", "perros", "compania"] if term in q_lower)
                        else 0
                    )
                    diff = abs(p_num - base_page)
                    return (-doc_boost, diff)

                page_scroll.sort(key=page_priority)

                for item in page_scroll:
                    chunk = self._payload_to_chunk(item.payload)
                    if chunk.chunk_id not in seen_ids:
                        seen_ids.add(chunk.chunk_id)
                        combined.append(chunk)
                logger.info(f"Filtro por página {base_page}: recuperados {len(page_scroll)} chunks prioritarios")
            except Exception as e:
                logger.warning(f"Búsqueda por filtro de página falló: {e}")

        # ── 2. BÚSQUEDA SEMÁNTICA (EMBEDDINGS DENSOS) ──────────────────────────
        try:
            query_vector = await self.llm.embed(query)
            semantic_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k * 2,
                with_payload=True,
                score_threshold=0.3,
            )
            for result in semantic_results:
                chunk = self._payload_to_chunk(result.payload)
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    combined.append(chunk)
        except Exception as e:
            logger.warning(f"Búsqueda semántica falló: {e}")

        # ── 3. BÚSQUEDA POR PALABRAS CLAVE (LÉXICA / FULL-TEXT) ─────────────────
        if not self._in_memory:
            cleaned_words = [
                re.sub(r'[^\w\d]', '', w.lower())
                for w in query.split()
            ]
            query_words = [
                w for w in cleaned_words
                if len(w) >= 2 and w not in STOPWORDS
            ]
            if query_words:
                try:
                    scroll_results, _ = await self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=models.Filter(
                            should=[
                                models.FieldCondition(
                                    key="text",
                                    match=models.MatchText(text=word),
                                )
                                for word in query_words[:4]
                            ]
                        ),
                        limit=top_k,
                        with_payload=True,
                    )
                    for result in scroll_results:
                        chunk = self._payload_to_chunk(result.payload)
                        if chunk.chunk_id not in seen_ids:
                            seen_ids.add(chunk.chunk_id)
                            combined.append(chunk)
                except Exception as e:
                    logger.warning(f"Búsqueda por palabras clave falló: {e}")

        return combined[: max(top_k, 6 if page_matches else top_k)]

    async def delete_by_document(self, document_id: str) -> None:
        """Elimina todos los chunks de un documento por su ID."""
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )]
                )
            ),
        )

    async def list_documents(self) -> list[dict]:
        """
        Lista todos los documentos indexados en la colección con métricas agregadas.
        """
        doc_stats = {}
        offset = None

        while True:
            try:
                records, next_offset = await self.client.scroll(
                    collection_name=self.collection_name,
                    limit=250,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                logger.error(f"Error listando documentos de Qdrant: {e}")
                break

            if not records:
                break

            for r in records:
                payload = r.payload or {}
                raw_name = payload.get("document_name", "Desconocido")
                clean_name = re.sub(r"^[0-9a-fA-F-]{36}_", "", raw_name)
                doc_key = clean_name

                if doc_key not in doc_stats:
                    doc_stats[doc_key] = {
                        "document_name": clean_name,
                        "raw_names": set(),
                        "document_id": payload.get("document_id", ""),
                        "total_chunks": 0,
                        "pages": set(),
                        "text_chunks": 0,
                        "table_chunks": 0,
                        "image_chunks": 0,
                        "images": set(),
                    }

                entry = doc_stats[doc_key]
                entry["raw_names"].add(raw_name)
                entry["total_chunks"] += 1
                page_num = payload.get("page_number")
                if page_num is not None:
                    entry["pages"].add(page_num)

                ctype = payload.get("chunk_type", "text")
                if ctype == "table":
                    entry["table_chunks"] += 1
                elif ctype == "image":
                    entry["image_chunks"] += 1
                else:
                    entry["text_chunks"] += 1

                img_path = payload.get("image_path")
                if img_path:
                    entry["images"].add(img_path)

            if next_offset is None:
                break
            offset = next_offset

        result = []
        for name, data in doc_stats.items():
            pages_list = sorted(list(data["pages"]))
            page_range = f"{min(pages_list)} - {max(pages_list)}" if pages_list else "1"
            result.append({
                "document_name": name,
                "document_id": data["document_id"],
                "total_chunks": data["total_chunks"],
                "total_pages": len(pages_list),
                "page_range": page_range,
                "text_chunks": data["text_chunks"],
                "table_chunks": data["table_chunks"],
                "image_chunks": data["image_chunks"],
                "total_images": len(data["images"]),
                "raw_names": list(data["raw_names"]),
            })

        result.sort(key=lambda x: x["document_name"].lower())
        return result

    async def delete_by_document_name(self, document_name: str) -> int:
        """
        Elimina todos los chunks que coincidan con document_name (exacto o con prefijo uuid_)
        y remueve del disco las imágenes extraídas correspondientes.
        """
        clean_target = re.sub(r"^[0-9a-fA-F-]{36}_", "", document_name).lower()
        matched_point_ids = []
        images_to_delete = set()
        offset = None

        while True:
            try:
                records, next_offset = await self.client.scroll(
                    collection_name=self.collection_name,
                    limit=250,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                logger.error(f"Error buscando puntos para eliminar: {e}")
                break

            if not records:
                break

            for r in records:
                payload = r.payload or {}
                raw_name = payload.get("document_name", "")
                clean_name = re.sub(r"^[0-9a-fA-F-]{36}_", "", raw_name).lower()

                if raw_name == document_name or clean_name == clean_target or raw_name.endswith(f"_{document_name}"):
                    matched_point_ids.append(r.id)
                    img = payload.get("image_path")
                    if img:
                        images_to_delete.add(img)

            if next_offset is None:
                break
            offset = next_offset

        if not matched_point_ids:
            logger.info(f"No se encontraron chunks para eliminar del documento '{document_name}'")
            return 0

        # Eliminar puntos de Qdrant en lotes
        batch_size = 500
        for i in range(0, len(matched_point_ids), batch_size):
            batch = matched_point_ids[i:i + batch_size]
            try:
                await self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=batch),
                )
            except Exception as e:
                logger.error(f"Error eliminando lote de puntos de Qdrant: {e}")

        logger.info(f"Eliminados {len(matched_point_ids)} chunks de Qdrant para '{document_name}'")

        # Eliminar imágenes asociadas del disco
        for img_path in images_to_delete:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    logger.debug(f"Imagen eliminada del disco: {img_path}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar imagen del disco ({img_path}): {e}")

        return len(matched_point_ids)

    def _payload_to_chunk(self, payload: dict) -> DocumentChunk:
        """Convierte un payload de Qdrant a una entidad DocumentChunk."""
        bbox = None
        if "bbox_x0" in payload:
            bbox = BoundingBox(
                x0=payload["bbox_x0"], y0=payload["bbox_y0"],
                x1=payload["bbox_x1"], y1=payload["bbox_y1"],
            )
        return DocumentChunk(
            chunk_id=payload.get("chunk_id", str(uuid.uuid4())),
            document_id=payload.get("document_id", ""),
            document_name=payload.get("document_name", ""),
            page_number=payload.get("page_number", 0),
            chunk_type=ChunkType(payload.get("chunk_type", "text")),
            text=payload.get("text", ""),
            image_path=payload.get("image_path"),
            bounding_box=bbox,
            nearest_image_path=payload.get("nearest_image_path"),
        )
