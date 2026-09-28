# ═══════════════════════════════════════════════════════════════
# DOCKERFILE — Frontend Streamlit
# También usa context: . (raíz del proyecto)
# ═══════════════════════════════════════════════════════════════

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar e instalar dependencias del frontend
COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del frontend
COPY frontend/ ./

# Streamlit no necesita carpeta de storage (solo lectura via volumen)
EXPOSE 8501

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
