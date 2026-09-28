"""
ADAPTADOR OLLAMA — Modelo de Lenguaje LOCAL (gratuito)

Implementa el mismo puerto LLMPort pero usando Ollama.
Ollama corre modelos como Llama 3.2 en tu propia máquina, sin costo.

IMPORTANTE para la sustentación:
Demostrar que puedes cambiar de proveedor de LLM SIN tocar la lógica de negocio
es exactamente lo que evalúan en la sección de Inyección de Dependencias.
Solo cambias la variable LLM_PROVIDER en el .env y el sistema usa otro modelo.
"""
import base64
import logging
import os
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.domain.ports import LLMPort

logger = logging.getLogger(__name__)


class OllamaAdapter(LLMPort):
    """
    Adaptador para Ollama — modelos de lenguaje corriendo localmente.
    Usa la API REST de Ollama que es compatible con la interfaz de OpenAI.
    """

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        logger.info(f"Ollama adapter inicializado: url={base_url}, modelo={model}")

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[str] | None = None,
    ) -> str:
        user_msg: dict = {"role": "user", "content": user_prompt}
        if images:
            b64_imgs = []
            for img_p in images:
                if img_p and os.path.exists(img_p):
                    try:
                        with open(img_p, "rb") as f:
                            b64_imgs.append(base64.b64encode(f.read()).decode("utf-8"))
                    except Exception as e:
                        logger.warning(f"Error cargando imagen para Ollama {img_p}: {e}")
            if b64_imgs:
                user_msg["images"] = b64_imgs

        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    user_msg,
                ],
                "options": {
                    "num_predict": 450,
                    "temperature": 0.2,
                },
                "stream": False,
            }
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            # Si el modelo Ollama no soporta visión (ej. llama3.2 texto), reintentar sin imágenes
            if response.status_code == 400 and "images" in user_msg:
                logger.info("Modelo Ollama no soporta visión directa. Reintentando consulta textual...")
                payload["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )

            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

    async def describe_image(self, image_path: str) -> str:
        """Describe una imagen usando Ollama si el modelo soporta visión."""
        if not image_path or not os.path.exists(image_path):
            return ""
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{
                            "role": "user",
                            "content": "Describe este esquema o diagrama técnico brevemente en español.",
                            "images": [b64],
                        }],
                        "stream": False,
                    },
                )
                if resp.status_code == 200:
                    return resp.json()["message"]["content"]
        except Exception as e:
            logger.warning(f"Ollama describe_image no disponible o modelo sin visión: {e}")
        return ""

    async def embed(self, text: str) -> list[float]:
        """
        Genera embeddings con Ollama.
        Nota: Para producción se recomienda usar nomic-embed-text con Ollama.
        """
        text = text[:4000]  # Ollama tiene límites menores
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {"model": "nomic-embed-text", "input": text}
            response = await client.post(
                f"{self.base_url}/api/embed",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
