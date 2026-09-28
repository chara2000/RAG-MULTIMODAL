"""
ADAPTADOR DE PARSEO PDF — PyMuPDF

Este es el "traductor" entre el mundo de los PDFs y nuestro dominio.
Implementa el puerto DocumentParser usando la librería PyMuPDF (fitz).

QUÉ HACE:
1. Abre el PDF página por página
2. Extrae texto con sus coordenadas (bounding boxes)
3. Extrae imágenes y las guarda en disco como archivos PNG
4. Agrupa el texto en chunks semánticos (por bloques, no por caracteres)
5. Para cada chunk de texto, encuentra la imagen más cercana en la página
   (esto es el "mapeo espacial" que pide la prueba)

CHUNKING SEMÁNTICO vs CHUNKING SIMPLE:
- Simple: "Corta cada 500 caracteres" → rompe frases y contexto
- Semántico: "Respeta la estructura del documento (títulos, párrafos, bloques)"
  → los chunks tienen sentido completo
"""
import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import pymupdf as fitz  # PyMuPDF — "fitz" era el nombre antiguo, ahora es "pymupdf"

from app.domain.entities import BoundingBox, ChunkType, DocumentChunk
from app.domain.ports import DocumentParser

logger = logging.getLogger(__name__)

# Tamaño mínimo de un chunk de texto (evita fragmentos muy pequeños)
MIN_CHUNK_CHARS = 100
# Tamaño máximo antes de forzar un corte
MAX_CHUNK_CHARS = 1500


class PyMuPDFParser(DocumentParser):
    """
    Implementación concreta del parser usando PyMuPDF.
    Extrae texto, tablas (detectadas por estructura) e imágenes.
    """

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        # Directorio donde guardaremos las imágenes extraídas
        self.images_dir = Path(storage_dir) / "extracted_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    async def parse(
        self, file_path: str, document_id: str, document_name: str
    ) -> list[DocumentChunk]:
        """
        Parsea el PDF de forma asíncrona.
        Usa run_in_executor para no bloquear el event loop de asyncio
        (el parseo de PDFs es CPU-intensive, no IO-bound).
        """
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None,  # Usa el ThreadPoolExecutor por defecto
            self._parse_sync,
            file_path,
            document_id,
            document_name,
        )
        return chunks

    def _parse_sync(
        self, file_path: str, document_id: str, document_name: str
    ) -> list[DocumentChunk]:
        """Parseo síncrono — se ejecuta en un hilo separado."""
        chunks: list[DocumentChunk] = []

        try:
            doc = fitz.open(file_path)
            logger.info(f"Parseando PDF: {document_name} ({doc.page_count} páginas)")

            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_number = page_num + 1  # Los humanos cuentan desde 1

                # 1. Extraer imágenes de la página
                page_images = self._extract_page_images(
                    doc, page, page_number, document_id
                )

                # 2. Extraer tablas estructuradas (TableFinder de PyMuPDF)
                page_tables = self._extract_page_tables(
                    page, page_number, document_id, document_name, page_images
                )

                # 3. Extraer texto en bloques (respetando el layout y evitando duplicar tablas)
                table_bboxes = [t.bounding_box for t in page_tables if t.bounding_box]
                text_chunks = self._extract_text_chunks(
                    page, page_number, document_id, document_name, page_images, table_bboxes
                )

                # 4. Consolidar todos los chunks de la página
                chunks.extend(page_images)
                chunks.extend(page_tables)
                chunks.extend(text_chunks)

            doc.close()
            logger.info(f"Extracción completa: {len(chunks)} chunks generados")
            return chunks

        except Exception as e:
            logger.error(f"Error parseando {file_path}: {e}")
            raise

    def _extract_page_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_number: int,
        document_id: str,
    ) -> list[DocumentChunk]:
        """
        Extrae imágenes incrustadas en la página.
        Las guarda como archivos PNG y crea chunks de tipo IMAGE.
        """
        image_chunks: list[DocumentChunk] = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]  # Referencia interna de PyMuPDF
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Guardar imagen en disco
                image_filename = f"{document_id}_p{page_number}_img{img_index}.{image_ext}"
                image_path = self.images_dir / image_filename
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Obtener la posición de la imagen en la página
                img_rect = page.get_image_rects(xref)
                bbox = None
                if img_rect:
                    r = img_rect[0]
                    bbox = BoundingBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)

                page_text_sample = " ".join(page.get_text().split()[:40]).strip()
                if page_text_sample:
                    text_content = f"[Diagrama/Imagen en página {page_number}]: {page_text_sample}"
                else:
                    text_content = f"[Imagen en página {page_number}, índice {img_index}]"

                chunk = DocumentChunk(
                    document_id=document_id,
                    document_name=os.path.basename(doc.name) if doc.name else "unknown",
                    page_number=page_number,
                    chunk_type=ChunkType.IMAGE,
                    text=text_content,
                    image_path=str(image_path),
                    bounding_box=bbox,
                )
                image_chunks.append(chunk)

            except Exception as e:
                logger.warning(f"No se pudo extraer imagen {img_index} de página {page_number}: {e}")

        return image_chunks

    def _extract_page_tables(
        self,
        page: fitz.Page,
        page_number: int,
        document_id: str,
        document_name: str,
        page_images: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """
        Extrae tablas estructuradas usando el TableFinder nativo de PyMuPDF.
        Convierte cada tabla en Markdown estructurado (con pipes | y cabeceras).
        """
        table_chunks: list[DocumentChunk] = []
        try:
            tabs = page.find_tables()
            for tab in tabs.tables:
                md_text = tab.to_markdown()
                if not md_text or len(md_text.strip()) < 20:
                    continue

                bbox = BoundingBox(
                    x0=tab.bbox[0],
                    y0=tab.bbox[1],
                    x1=tab.bbox[2],
                    y1=tab.bbox[3],
                )
                nearest_img = self._find_nearest_image(bbox, page_images)
                table_chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_number,
                        chunk_type=ChunkType.TABLE,
                        text=f"[Tabla Técnica - Página {page_number}]:\n{md_text}",
                        bounding_box=bbox,
                        nearest_image_path=nearest_img,
                    )
                )
        except Exception as e:
            logger.warning(f"Error extrayendo tablas en página {page_number}: {e}")

        return table_chunks

    def _extract_text_chunks(
        self,
        page: fitz.Page,
        page_number: int,
        document_id: str,
        document_name: str,
        page_images: list[DocumentChunk],
        table_bboxes: Optional[list[BoundingBox]] = None,
    ) -> list[DocumentChunk]:
        """
        Extrae texto usando el layout del documento.
        PyMuPDF provee bloques de texto con sus coordenadas — usamos eso
        para hacer chunking semántico (respetando párrafos y secciones).
        """
        chunks: list[DocumentChunk] = []
        # get_text("blocks") retorna lista de: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type=0 es texto, block_type=1 es imagen
        blocks = page.get_text("blocks", sort=True)

        current_text = ""
        current_bbox: Optional[BoundingBox] = None

        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block

            # Solo procesamos bloques de texto (tipo 0)
            if block_type != 0:
                continue

            text = text.strip()
            if not text:
                continue

            # Si el bloque de texto cae dentro de una tabla ya extraída, no duplicar como texto plano
            if table_bboxes:
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                if any(tb.x0 <= cx <= tb.x1 and tb.y0 <= cy <= tb.y1 for tb in table_bboxes):
                    continue

            # Determinar si es una tabla (heurística de respaldo)
            chunk_type = self._detect_chunk_type(text)

            # Estrategia de chunking semántico:
            # Si el chunk actual + nuevo texto excede el máximo, guarda el actual
            if current_text and len(current_text) + len(text) > MAX_CHUNK_CHARS:
                if len(current_text) >= MIN_CHUNK_CHARS:
                    nearest_img = self._find_nearest_image(current_bbox, page_images)
                    chunks.append(DocumentChunk(
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_number,
                        chunk_type=ChunkType.TEXT,
                        text=current_text,
                        bounding_box=current_bbox,
                        nearest_image_path=nearest_img,
                    ))
                current_text = text
                current_bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
            else:
                # Acumular texto en el chunk actual
                current_text = (current_text + "\n" + text).strip() if current_text else text
                if current_bbox is None:
                    current_bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
                else:
                    # Expandir el bounding box para cubrir ambos bloques
                    current_bbox = BoundingBox(
                        x0=min(current_bbox.x0, x0),
                        y0=min(current_bbox.y0, y0),
                        x1=max(current_bbox.x1, x1),
                        y1=max(current_bbox.y1, y1),
                    )

        # No olvidar el último chunk acumulado
        if current_text and len(current_text) >= MIN_CHUNK_CHARS:
            nearest_img = self._find_nearest_image(current_bbox, page_images)
            chunks.append(DocumentChunk(
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                chunk_type=ChunkType.TEXT,
                text=current_text,
                bounding_box=current_bbox,
                nearest_image_path=nearest_img,
            ))

        return chunks

    def _detect_chunk_type(self, text: str) -> ChunkType:
        """
        Heurística para detectar si un bloque de texto es una tabla.
        Las tablas suelen tener muchas tabulaciones o caracteres de separación.
        """
        tab_count = text.count("\t")
        pipe_count = text.count("|")
        lines = text.split("\n")
        if tab_count > 3 or pipe_count > 3:
            return ChunkType.TABLE
        return ChunkType.TEXT

    def _find_nearest_image(
        self,
        text_bbox: Optional[BoundingBox],
        page_images: list[DocumentChunk],
    ) -> Optional[str]:
        """
        Busca la imagen más cercana al bloque de texto en la misma página.
        Usa la distancia euclidiana entre los centros de los bounding boxes.

        Esto es clave para el contexto visual: cuando el texto menciona
        "ver diagrama X", la imagen más cercana probablemente ES ese diagrama.
        """
        if not text_bbox or not page_images:
            return None

        # Centro del bloque de texto
        text_cx = (text_bbox.x0 + text_bbox.x1) / 2
        text_cy = (text_bbox.y0 + text_bbox.y1) / 2

        min_dist = float("inf")
        nearest_path: Optional[str] = None

        for img_chunk in page_images:
            if not img_chunk.bounding_box or not img_chunk.image_path:
                continue

            # Centro de la imagen
            img_cx = (img_chunk.bounding_box.x0 + img_chunk.bounding_box.x1) / 2
            img_cy = (img_chunk.bounding_box.y0 + img_chunk.bounding_box.y1) / 2

            dist = ((text_cx - img_cx) ** 2 + (text_cy - img_cy) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_path = img_chunk.image_path

        return nearest_path
