"""
ENDPOINTS DE GESTIÓN DE DOCUMENTOS (CRUD)

GET    /documents            → Lista todos los documentos indexados con métricas
DELETE /documents/{name}     → Elimina un documento y sus vectores/imágenes
"""
import logging
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_vector_store
from app.domain.ports import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Gestión de Documentos"])


@router.get("/", summary="Listar todos los documentos indexados")
async def list_documents(vector_store: VectorStore = Depends(get_vector_store)):
    """
    Retorna la lista completa de documentos disponibles en el sistema con
    sus estadísticas agregadas: total de fragmentos, páginas, tablas e imágenes.
    """
    try:
        docs = await vector_store.list_documents()
        return {"documents": docs, "total_documents": len(docs)}
    except Exception as e:
        logger.error(f"Error listando documentos: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo documentos: {str(e)}")


@router.delete("/{document_name:path}", summary="Eliminar un documento indexado")
async def delete_document(
    document_name: str,
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    Elimina todos los fragmentos vectoriales, metadatos y diagramas/imágenes
    asociados a un documento específico.
    """
    try:
        deleted_count = await vector_store.delete_by_document_name(document_name)
        if deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró el documento '{document_name}' para eliminar.",
            )
        return {
            "deleted": True,
            "document_name": document_name,
            "points_deleted": deleted_count,
            "message": f"Documento '{document_name}' y sus recursos asociados fueron eliminados.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando documento '{document_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Error al eliminar documento: {str(e)}")
