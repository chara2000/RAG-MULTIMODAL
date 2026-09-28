"""
CONFIGURACIÓN DEL SISTEMA

Lee variables de entorno del archivo .env de forma tipada y validada.

NOTA SOBRE QDRANT:
Cambiamos de QDRANT_URL a QDRANT_HOST + QDRANT_PORT porque en docker-compose
cada variable se define por separado (mejor para lectura y sobreescritura).

MODO SIN DOCKER:
Si QDRANT_HOST=localhost y el puerto no responde, el adaptador Qdrant
hace fallback automático a in-memory. No necesitas instalar nada extra.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Entorno de Ejecución ---
    environment: str = "development"  # "development", "production", "test"
    debug: bool = False

    # --- OpenAI ---
    openai_api_key: str = "sk-fake-key-for-testing"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- LLM Provider: "openai" o "ollama" ---
    llm_provider: str = "openai"

    # --- Ollama (modelo local gratuito) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # --- Qdrant: host y puerto por separado (como en docker-compose) ---
    qdrant_host: str = "localhost"    # En Docker: "qdrant" (nombre del servicio)
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_documents"

    # --- Almacenamiento de archivos y imágenes extraídas ---
    # En Docker: /app/storage (volumen storage_data)
    # En local:  ./storage  (carpeta local)
    storage_dir: str = "./storage"

    # --- Logging ---
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton de configuración — se lee el .env una sola vez."""
    return Settings()
