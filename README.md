# 🔍 RAG Multimodal — Asistente de Inteligencia Técnica y Visión

Sistema RAG (*Retrieval-Augmented Generation*) para ingesta, extracción espacial, indexación híbrida y consulta en lenguaje natural sobre documentos PDF técnicos complejos que contienen texto, tablas estructuradas, esquemas y diagramas de circuitos.

Diseñado bajo principios de **Arquitectura Hexagonal (Puertos y Adaptadores)** y buenas prácticas de ingeniería de software listo para producción.

---

## 🏛️ Arquitectura del Sistema

El proyecto sigue una arquitectura limpia desacoplando por completo el transporte HTTP, los casos de uso del negocio y las integraciones con servicios externos (LLM, base de datos vectorial y parser).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CLIENTE FRONTEND                                │
│                   (Streamlit en puerto :8501)                          │
│  • Pestañas flotantes: Charlar, Documentos (CRUD), Historial, Sistema  │
│  • Conmutador dinámico de proveedor: OpenAI GPT-4o vs Ollama Local     │
│  • Visualizador de esquemas/diagramas recuperados y fuentes exactas    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY (FastAPI :8000)                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Capa de Transporte / Endpoints:                                  │  │
│  │   • POST /ingest/        • GET /ingest/{id}                      │  │
│  │   • POST /query          • GET /documents   • DELETE /documents  │  │
│  └────────────────────────────────┬─────────────────────────────────┘  │
│  ┌────────────────────────────────┴─────────────────────────────────┐  │
│  │ Capa de Aplicación (Casos de Uso):                               │  │
│  │   • IngestDocumentUseCase (Extracción asíncrona en background)   │  │
│  │   • AnswerQueryUseCase    (Recuperación híbrida + Visión LLM)    │  │
│  └────────────────────────────────┬─────────────────────────────────┘  │
│  ┌────────────────────────────────┴─────────────────────────────────┐  │
│  │ Capa de Dominio (Pura, sin dependencias de frameworks):          │  │
│  │   • Entidades: DocumentChunk, IngestionJob, QueryResult, BBox   │  │
│  │   • Puertos / Contratos: DocumentParser, VectorStore, LLMPort   │  │
│  └────────────────────────────────┬─────────────────────────────────┘  │
│  ┌────────────────────────────────┴─────────────────────────────────┐  │
│  │ Capa de Infraestructura (Adaptadores Concretos):                 │  │
│  │   • PyMuPDFParser        (Mapeo espacial de bounding boxes)      │  │
│  │   • QdrantAdapter        (Búsqueda híbrida vectorial + texto)    │  │
│  │   • OpenAIAdapter        (GPT-4o-mini Vision + Retry Tenacity)   │  │
│  │   • OllamaAdapter        (Llama 3.2 local sin costo en Docker)   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└───────────────────┬─────────────────────────────────┬──────────────────┘
                    │                                 │
                    ▼                                 ▼
         ┌─────────────────────┐           ┌─────────────────────┐
         │ Qdrant (:6333/:6334)│           │  Ollama (:11434)    │
         │  Base de Datos      │           │  Llama 3.2 / Nomic  │
         │  Vectorial Híbrida  │           │  Ejecución Local    │
         └─────────────────────┘           └─────────────────────┘
```

---

## 📋 Registro de Decisiones Técnicas

| Decisión Arquitectónica | Tecnología / Estrategia | Justificación de Ingeniería |
| :--- | :--- | :--- |
| **Arquitectura de Software** | Hexagonal (Ports & Adapters) | Aísla la lógica de negocio del framework web y de proveedores externos. Permite cambiar de Qdrant a Milvus o de OpenAI a Ollama sin alterar los casos de uso. |
| **Framework Backend** | FastAPI (Python 3.11) | Rendimiento asíncrono nativo, validación automática mediante Pydantic y documentación OpenAPI interactiva en `/docs`. |
| **Procesamiento Asíncrono** | `BackgroundTasks` + Pool de Hilos | La extracción y rasterizado de PDFs es intensiva en CPU. Ejecutarla en segundo plano con `run_in_executor` evita bloquear el bucle de eventos (`event loop`) de la API. |
| **Mapeo Espacial Multimodal** | PyMuPDF (`fitz`) + Centroides Bounding Boxes | Extrae coordenadas `(x0, y0, x1, y1)` de cada bloque de texto, tabla e imagen. Asocia cada párrafo con el esquema más cercano calculando distancia euclidiana entre centroides. |
| **Estrategia de Chunking** | Semántico por bloques y tablas | No corta ciegamente por conteo fijo de caracteres. Mantiene juntas secciones con sentido y formatea tablas completas en Markdown con pipes (`\|`). |
| **Base de Datos Vectorial** | Qdrant (Contenerizado) | Motor de búsqueda de alta eficiencia con soporte nativo de filtros por metadatos, búsqueda vectorial por similitud de coseno y búsqueda híbrida. |
| **Resiliencia ante Fallos** | Tenacity (Exponential Backoff) | Gestiona de forma automática caídas transitorias de red y límites de tasa (*rate limits*) de la API del LLM reintentando progresivamente (1s, 2s, 4s... hasta 30s). |
| **Generación Defensiva** | Prompt Engineering Defensivo | Instrucción estricta al modelo para admitir explícitamente cuando el contexto no contiene información suficiente, mitigando alucinaciones técnicas. |
| **Frontend / Cliente** | Streamlit con UI Avanzada | Interfaz con pestañas flotantes estilo píldora, entrada de texto fija al pie con compensación anti-traslape y soporte de traducción sin rotura de DOM en React. |

---

## 🚀 Inicio Rápido (Despliegue con Docker)

### Requisitos Previos
* Docker y Docker Compose instalados en el sistema.
* Clave de API de OpenAI (para inferencia en la nube) o el servicio local de Ollama incluido en el compose.

### 1. Clonación y Configuración del Entorno
```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd rag-multimodal

# Crear archivo de variables de entorno a partir de la plantilla
cp .env.example .env
```

Edita el archivo `.env` para colocar tu API Key:
```env
OPENAI_API_KEY=sk-proj-tu-api-key-aqui
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
```

### 2. Levantar el Stack Completo
Ejecuta el siguiente comando para construir e iniciar los 4 servicios orquestados:

```bash
docker compose up -d --build
```

### 3. Servicios Disponibles

| Servicio | URL Local | Descripción |
| :--- | :--- | :--- |
| **Frontend Web** | [http://localhost:8501](http://localhost:8501) | Interfaz de chat interactivo, gestión de documentos y métricas |
| **API Backend** | [http://localhost:8000/docs](http://localhost:8000/docs) | Documentación interactiva Swagger UI / OpenAPI |
| **Vector DB (Qdrant)** | [http://localhost:6333/dashboard](http://localhost:6333/dashboard) | Consola visual de colecciones y puntos vectoriales |
| **Ollama Local** | [http://localhost:11434](http://localhost:11434) | Motor de inferencia local para modelos de código abierto |

---

## 🧪 Ejecución de Pruebas Automatizadas

El sistema cuenta con una suite completa de pruebas unitarias e integradas aislando componentes mediante **Mocks** (`AsyncMock`):

```bash
# Ejecutar toda la suite dentro del contenedor backend
docker exec rag_backend pytest tests/ -v

# Ejecutar con reporte de cobertura
docker exec rag_backend pytest --cov=app tests/
```

### Cobertura de Pruebas:
* `tests/unit/test_answer_query.py`: Valida recuperación híbrida, deduplicación de fuentes, inclusión de imágenes y respuesta defensiva ante falta de contexto.
* `tests/unit/test_ingest_document.py`: Valida ciclo de vida del trabajo de ingesta (`pending` ➔ `processing` ➔ `completed` / `failed`) y desacople del parser.
* `tests/integration/test_api.py`: Valida endpoints HTTP de subida, consulta, listado y eliminación con cliente de pruebas de FastAPI.

---

## ⚡ Prueba de Estrés: Ingesta de 100 Manuales Técnicos

Para validar cómo responde el sistema ante un volumen masivo de documentos técnicos sin bloquear la API:

1. Ejecuta el script de descarga concurrente provisto:
   ```bash
   python descargar_100_manuales.py
   ```
   *(Descarga automáticamente 100 manuales técnicos ligeros reales desde el repositorio público de especificaciones de GitHub `tpn/pdfs`).*

2. En la interfaz web ([http://localhost:8501](http://localhost:8501)), ingresa a la pestaña **Documentos**.
3. Selecciona los 100 archivos en la carpeta `lote_100_manuales/` y arrástralos en lote (*drag & drop*).
4. El backend procesará los documentos concurrentemente en segundo plano con seguimiento en tiempo real del `job_id`.

---

## 📡 Referencia de Endpoints Principales

### 1. Ingesta de Documentos
* **`POST /ingest/`**: Recibe un archivo PDF en `multipart/form-data`. Devuelve inmediatamente `job_id` y estado `pending`.
* **`GET /ingest/{job_id}`**: Devuelve el progreso del procesamiento (`pending`, `processing`, `completed`, `failed`), total de chunks generados y mensajes de estado.

### 2. Consulta y RAG
* **`POST /query`**:
  ```json
  {
    "question": "¿Cómo se conecta la alimentación y qué circuito se utiliza para la entrada de energía y USB?",
    "top_k": 5,
    "provider": "openai"
  }
  ```
  Devuelve la respuesta sintetizada, fuentes citadas con página exacta (`sources`), imágenes asociadas (`image_urls`) y bandera de suficiencia de contexto (`has_sufficient_context`).

### 3. Biblioteca y CRUD
* **`GET /documents`**: Lista los documentos indexados, número de páginas, chunks de texto, tablas e imágenes.
* **`DELETE /documents/{document_name}`**: Elimina de manera atómica todos los vectores asociados a un manual en Qdrant y sus imágenes en almacenamiento.

---

## 🛡️ Principios de Resiliencia y Mitigación de Alucinaciones
1. **Deduplicación Estricta:** El sistema unifica fragmentos repetidos y filtra diagramas con firmas de imagen idénticas para evitar saturar el contexto del modelo.
2. **Citas de Página Infalibles:** El prompt de sistema obliga al modelo a citar el número de página exacto reportado en los metadatos de los fragmentos recuperados.
3. **Respuesta Defensiva:** Si la similitud de los chunks recuperados es nula o insuficiente para contestar con certeza técnica, el sistema responde explícitamente notificando la falta de información en lugar de especular.
