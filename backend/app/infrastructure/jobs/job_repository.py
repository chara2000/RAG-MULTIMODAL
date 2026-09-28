"""
REPOSITORIO DE JOBS — En memoria (Thread-safe)

Guarda el estado de los trabajos de ingesta en RAM.
En producción usaríamos Redis o una base de datos, pero para esta prueba
una estructura en memoria con threading.Lock es suficiente y más simple.

El Lock garantiza que múltiples requests simultáneos no corrompan los datos.
"""
import threading
from typing import Optional

from app.domain.entities import IngestionJob
from app.domain.ports import JobRepository


class InMemoryJobRepository(JobRepository):
    """
    Repositorio de jobs en memoria.
    Thread-safe gracias al Lock de Python.
    """

    def __init__(self):
        self._store: dict[str, IngestionJob] = {}
        self._lock = threading.Lock()

    def save(self, job: IngestionJob) -> None:
        """Guarda o actualiza un job. El Lock evita condiciones de carrera."""
        with self._lock:
            self._store[job.job_id] = job

    def get(self, job_id: str) -> Optional[IngestionJob]:
        """Busca un job por ID. Retorna None si no existe."""
        with self._lock:
            return self._store.get(job_id)

    def list_all(self) -> list[IngestionJob]:
        """Lista todos los jobs (útil para debugging)."""
        with self._lock:
            return list(self._store.values())
