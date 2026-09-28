"""
ADAPTADOR OPENAI — LLM + Embeddings

Implementa el puerto LLMPort usando la API de OpenAI.

RESILIENCIA con Tenacity:
El decorador @retry aplica "Exponential Backoff" — si OpenAI falla
(por límite de tasa o error de red), espera 1s, luego 2s, luego 4s, etc.
Esto evita que el sistema colapse en cascada.

PATRÓN DEFENSIVO:
El prompt de sistema le ordena al LLM que admita cuando no tiene información.
Esto mitiga las "alucinaciones" (respuestas inventadas).
"""
import logging
import base64
import os
from typing import Optional

from openai import AsyncOpenAI, RateLimitError, APIConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.domain.ports import LLMPort

logger = logging.getLogger(__name__)

# Tipos de error que merecen un reintento (errores transitorios de red/rate limit)
RETRYABLE_ERRORS = (RateLimitError, APIConnectionError)


class OpenAIAdapter(LLMPort):
    """
    Adaptador para OpenAI GPT (con soporte Multimodal Vision).
    Si necesitas cambiar a Anthropic Claude u otro, solo creas otro adaptador
    que implemente LLMPort — el resto del código no cambia nada.
    """

    def __init__(self, api_key: str, model: str, embedding_model: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model
        logger.info(f"OpenAI adapter inicializado: modelo={model}")

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=1, min=1, max=30),  # Espera: 1s, 2s, 4s, 8s...
        stop=stop_after_attempt(4),  # Máximo 4 intentos
        reraise=True,  # Si falla los 4, propaga la excepción
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[list[str]] = None,
    ) -> str:
        """
        Llama a GPT con el prompt y opcionalmente imágenes en base64 (Multimodal).
        Reintentos automáticos si hay rate limit o error de red.
        """
        logger.debug(f"Llamando a OpenAI [{self.model}] multimodal={bool(images)}...")

        # Construir contenido del mensaje de usuario
        if images:
            user_content = [{"type": "text", "text": user_prompt}]
            for img_path in images[:3]:  # Hasta 3 imágenes relevantes
                if img_path and os.path.exists(img_path):
                    try:
                        ext = os.path.splitext(img_path)[1].lower().replace(".", "")
                        mime = "image/png" if ext == "png" else "image/jpeg"
                        with open(img_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}
                        })
                    except Exception as e:
                        logger.warning(f"No se pudo cargar imagen {img_path} para el prompt: {e}")
        else:
            user_content = user_prompt

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,  # Temperatura baja = respuestas más deterministas
            max_tokens=1500,
        )
        answer = response.choices[0].message.content or ""
        logger.debug(f"Respuesta OpenAI recibida ({len(answer)} chars)")
        return answer

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(3),
        reraise=False,
    )
    async def describe_image(self, image_path: str) -> str:
        """
        Genera una descripción técnica concisa de una imagen, diagrama o esquema.
        Esto permite indexar elementos visuales en el vector store con significado semántico.
        """
        if not image_path or not os.path.exists(image_path):
            return ""

        try:
            ext = os.path.splitext(image_path)[1].lower().replace(".", "")
            mime = "image/png" if ext == "png" else "image/jpeg"
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Describe técnicamente qué muestra esta página o diagrama técnico "
                                    "(título principal, componentes señalados, indicadores, controles y función). "
                                    "Sé conciso y menciona palabras clave clave exactas en español."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                        ],
                    }
                ],
                max_tokens=200,
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Error generando descripción visual para {image_path}: {e}")
            return ""

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def embed(self, text: str) -> list[float]:
        """
        Convierte texto en vector numérico usando el modelo de embeddings.
        text-embedding-3-small produce vectores de 1536 dimensiones.
        """
        # Truncar texto muy largo para no exceder el límite del modelo
        text = text[:8000]
        response = await self.client.embeddings.create(
            input=text,
            model=self.embedding_model,
        )
        return response.data[0].embedding
