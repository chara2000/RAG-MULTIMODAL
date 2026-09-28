"""
Backend del Sistema RAG Multimodal.
Arquitectura Hexagonal por capas:
- domain/    → Entidades y contratos (sin dependencias externas)
- application/ → Casos de uso (orquesta la lógica)
- infrastructure/ → Adaptadores concretos (OpenAI, Qdrant, PyMuPDF)
- api/       → Capa HTTP (FastAPI)
"""
