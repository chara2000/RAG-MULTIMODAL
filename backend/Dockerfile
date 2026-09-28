# ═══════════════════════════════════════════════════════════════
# DOCKERFILE — Backend FastAPI
#
# IMPORTANTE: Este Dockerfile se construye con context: . (raíz)
# Eso significa que Docker puede ver TODA la carpeta del proyecto.
# Por eso copiamos con "backend/" como prefijo.
# ═══════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Instalar utilidades del sistema que necesita PyMuPDF
# y curl para los healthchecks de Docker
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar e instalar dependencias PRIMERO (aprovecha el cache de Docker)
# Si no cambias requirements.txt, Docker no reinstala todo en cada build
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente del backend
COPY backend/ ./

# Crear la carpeta de almacenamiento con subcarpeta de imágenes
RUN mkdir -p /app/storage/extracted_images

EXPOSE 8000

# Arrancar con uvicorn: app.api.main es el módulo Python, "app" es la instancia FastAPI
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
