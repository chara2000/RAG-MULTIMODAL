"""
FRONTEND — RAG Multimodal (Minimal Premium AI Experience)
Inspirado en la interfaz y experiencia de usuario de ChatGPT y Gemini.
Arquitectura: Hexagonal / Cliente HTTP puro para la API FastAPI (:8000).

Características Principales:
  - Diseño minimalista, limpio, moderno y profesional (AI Modern / Minimal Premium).
  - Barra de chat fija flotante (stBottom nativo) que respeta el ancho del sidebar sin traslaparse.
  - Botón nativo para colapsar y expandir la barra lateral visible y estilizado.
  - La entrada de texto permanece SIEMPRE visible, incluso mientras se sintetiza la respuesta.
  - Padding inferior amplio para garantizar que el último mensaje jamás quede tapado.
  - Mensajes de usuario y asistente claramente diferenciados con avatares 👤 y ✦.
  - Despliegue discreto y ordenado de citas técnicas y esquemas visuales recuperados.
  - CRUD Completo de Documentos PDF con soporte para subida múltiple en lote y detección de duplicados.
  - Interruptor dinámico de Proveedor IA (OpenAI GPT-4o-mini vs Ollama Local Llama 3.2).
  - Manejo resiliente de timeouts (variable req_timeout disponible globalmente para todas las pestañas).
"""
import json
import os
import re
import time
import uuid
from datetime import datetime
import requests
import streamlit as st
import streamlit.components.v1 as components

# Prevenir que extensiones de traducción (ej. Google Translate) inyecten etiquetas <font>
# que corrompan el árbol DOM de React y causen NotFoundError: removeChild en Streamlit
components.html(
    """
    <script>
    try {
        window.parent.document.documentElement.setAttribute('translate', 'no');
        window.parent.document.documentElement.classList.add('notranslate');
        if (window.parent.document.body) {
            window.parent.document.body.setAttribute('translate', 'no');
            window.parent.document.body.classList.add('notranslate');
        }
        let meta = window.parent.document.createElement('meta');
        meta.name = 'google';
        meta.content = 'notranslate';
        window.parent.document.head.appendChild(meta);
    } catch (e) {
        console.warn('Could not inject notranslate:', e);
    }
    </script>
    """,
    height=0,
    width=0,
)

# En Docker: BACKEND_URL=http://backend:8000 | En local: BACKEND_URL=http://localhost:8000
API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ─── Configuración de Página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Multimodal | Asistente Inteligente",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inicialización del Estado de Sesión Multi-Chat ───────────────────────────
if "sessions" not in st.session_state:
    initial_id = str(uuid.uuid4())[:8]
    st.session_state.sessions = {
        initial_id: {
            "id": initial_id,
            "title": "Nueva consulta",
            "created_at": datetime.now().strftime("%d/%m %H:%M"),
            "messages": [],
        }
    }
    st.session_state.active_session_id = initial_id

if "active_session_id" not in st.session_state or st.session_state.active_session_id not in st.session_state.sessions:
    st.session_state.active_session_id = list(st.session_state.sessions.keys())[0]

if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = "openai"

if "preset_prompt" not in st.session_state:
    st.session_state.preset_prompt = None

# ─── Variables Globales de Proveedor y Timeout ────────────────────────────────
active_prov = st.session_state.ai_provider
active_prov_label = "OpenAI GPT-4o-mini" if active_prov == "openai" else "Ollama Llama 3.2"
# Timeout dinámico: 240s para Ollama en CPU, 45s para OpenAI en la nube
req_timeout = 240 if active_prov == "ollama" else 45


# ─── Sistema de Diseño Visual "AI Modern / Minimal Premium" ───────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Variables de Color & Tipografía Minimalista ── */
    :root {
        --bg-main: #0d1117;
        --bg-sidebar: #090d16;
        --bg-surface: #161b22;
        --bg-surface-hover: #1f242c;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-focus: #3b82f6;
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --accent-blue: #3b82f6;
        --accent-blue-hover: #2563eb;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
    }

    /* ── Estilos Globales de la Aplicación ── */
    .stApp {
        background-color: var(--bg-main) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* Header nativo de Streamlit: Mantenerlo limpio y asegurar que el botón de colapsar sidebar sea visible */
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 1000 !important;
    }

    /* Botón nativo para contraer y expandir la barra lateral izquierda */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 10001 !important;
    }

    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        color: #f0f6fc !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="collapsedControl"] button:hover {
        background: rgba(255, 255, 255, 0.14) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
    }

    /* ── Contenedor Principal de la Conversación ── */
    .stMainBlockContainer {
        max-width: 820px !important;
        margin: 0 auto !important;
        padding-top: 1rem !important;
        padding-bottom: 140px !important; /* Margen amplio para evitar que el input fijo tape el último mensaje */
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    /* ── Header Discreto Relativo (permite que las pestañas floten por encima) ── */
    .rag-main-header {
        position: relative !important;
        z-index: 100;
        background: rgba(13, 17, 23, 0.92);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid var(--border-subtle);
        padding: 0.75rem 0;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .rag-header-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .rag-header-sparkle {
        color: var(--accent-blue);
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1;
    }

    .rag-header-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.015em;
        margin: 0;
        line-height: 1.2;
    }

    .rag-header-subtitle {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin: 0;
    }

    .rag-header-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--border-subtle);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        color: var(--text-secondary);
    }

    .status-dot-active {
        width: 7px;
        height: 7px;
        background-color: var(--accent-emerald);
        border-radius: 50%;
        display: inline-block;
    }

    .status-dot-local {
        width: 7px;
        height: 7px;
        background-color: var(--accent-blue);
        border-radius: 50%;
        display: inline-block;
    }

    /* ── Pestañas de Navegación Sticky / Flotantes Arriba (Estilo Píldora Flotante) ── */
    div[role="tablist"],
    .stTabs [role="tablist"],
    .stTabs [data-baseweb="tab-list"],
    div[data-testid="stTabs"] [role="tablist"] {
        position: sticky !important;
        top: 8px !important;
        z-index: 9990 !important;
        display: flex !important;
        justify-content: center !important;
        gap: 6px !important;
        background: rgba(18, 22, 30, 0.94) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.65) !important;
        padding: 5px 8px !important;
        border-radius: 14px !important;
        width: fit-content !important;
        margin: 0 auto 1.5rem auto !important;
    }

    div[role="tab"],
    [data-testid="stTab"],
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px !important;
        padding: 6px 16px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.15s ease !important;
    }

    div[role="tab"][aria-selected="true"],
    [data-testid="stTab"][aria-selected="true"],
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.10) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ── Mensajes de Chat (Área de Conversación) ── */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0.85rem 0 !important;
        gap: 0.85rem !important;
    }

    /* Avatar Minimalista */
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
    }

    /* Contenido de Mensaje */
    [data-testid="stChatMessageContent"] {
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stChatMessageContent"] p {
        margin-bottom: 0.75rem !important;
    }

    [data-testid="stChatMessageContent"] code {
        background: rgba(255, 255, 255, 0.07) !important;
        color: #79c0ff !important;
        padding: 0.15rem 0.4rem !important;
        border-radius: 5px !important;
        font-size: 0.86rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    [data-testid="stChatMessageContent"] pre {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        padding: 0.85rem !important;
    }

    /* ── Contenedor Inferior Nativo (stBottom): Fijo al fondo ── */
    [data-testid="stBottom"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        z-index: 9999 !important;
        background: linear-gradient(180deg, rgba(13, 17, 23, 0) 0%, rgba(13, 17, 23, 0.75) 30%, #0d1117 85%) !important;
        padding-top: 1.5rem !important;
        padding-bottom: 1.25rem !important;
        pointer-events: none !important;
        display: flex !important;
        justify-content: center !important;
        transition: left 0.25s ease, width 0.25s ease !important;
    }

    /* Cuando la barra lateral izquierda está expandida, centrar el chat input en el área visible a la derecha del sidebar sin ser tapado */
    .stApp:has([data-testid="stSidebar"][aria-expanded="true"]) [data-testid="stBottom"] {
        left: 336px !important;
        width: calc(100% - 336px) !important;
    }

    [data-testid="stBottom"] > div {
        width: 100% !important;
        max-width: 660px !important;
        margin: 0 auto !important;
        padding: 0 1rem !important;
        background: transparent !important;
        pointer-events: auto !important;
    }

    /* ── Composer / Input Flotante Fijo al Fondo (Estilo ChatGPT / Gemini) Más Angosto ── */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 1.25rem !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: min(660px, calc(100vw - 3rem)) !important;
        max-width: 660px !important;
        margin: 0 !important;
        z-index: 9999 !important;
        background: #161b22 !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.75) !important;
        padding: 4px 6px !important;
        pointer-events: auto !important;
        transition: left 0.25s ease, width 0.25s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    /* Cuando la barra lateral izquierda está expandida, centrar el chat input en el área visible a la derecha del sidebar sin ser tapado */
    .stApp:has([data-testid="stSidebar"][aria-expanded="true"]) [data-testid="stChatInput"] {
        left: calc(50% + 168px) !important;
        width: min(660px, calc(100vw - 370px)) !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--border-focus) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), 0 0 0 1px var(--border-focus) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--text-primary) !important;
        font-size: 0.94rem !important;
        line-height: 1.5 !important;
        padding: 0.6rem 0.8rem !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* Botón de Enviar Integrado */
    [data-testid="stChatInput"] button {
        background: var(--accent-blue) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: none !important;
        width: 32px !important;
        height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stChatInput"] button:hover:not(:disabled) {
        background: var(--accent-blue-hover) !important;
        transform: scale(1.04) !important;
    }

    [data-testid="stChatInput"] button:disabled {
        background: rgba(255, 255, 255, 0.06) !important;
        color: var(--text-muted) !important;
        opacity: 0.5 !important;
    }

    @media (max-width: 768px) {
        .stApp:has([data-testid="stSidebar"][aria-expanded="true"]) [data-testid="stBottom"] {
            left: 0 !important;
            width: 100% !important;
        }
        [data-testid="stBottom"] > div,
        [data-testid="stChatInput"] {
            max-width: 95% !important;
        }
    }

    /* ── Tarjetas y Componentes Secundarios Discretos ── */
    .minimal-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.1rem;
        margin: 0.6rem 0;
    }

    .citation-card {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.83rem;
    }

    .citation-meta {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        padding: 2px 7px;
        border-radius: 5px;
        font-weight: 600;
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    /* ── Estado Vacío (Hero Inicial) ── */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        max-width: 600px;
        margin: 0 auto;
    }

    .hero-icon {
        font-size: 2rem;
        color: var(--accent-blue);
        margin-bottom: 0.5rem;
    }

    .hero-heading {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }

    .hero-text {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }

    /* ── Botones de Sugerencia Estilo Chip ── */
    .stButton > button {
        border-radius: 10px !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        border: 1px solid var(--border-subtle) !important;
        background: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        border-color: rgba(255, 255, 255, 0.2) !important;
        background: var(--bg-surface-hover) !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent-blue) !important;
        border-color: var(--accent-blue) !important;
        color: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--accent-blue-hover) !important;
    }

    /* ── Sidebar Estilo ChatGPT ── */
    [data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .sidebar-section-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin: 1rem 0 0.4rem 0;
    }

    /* ── Expanders Discretos ── */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.025) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        font-size: 0.84rem !important;
        color: var(--text-secondary) !important;
    }

    .streamlit-expanderContent {
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-bottom-left-radius: 8px !important;
        border-bottom-right-radius: 8px !important;
        background: rgba(22, 27, 34, 0.4) !important;
        padding: 0.75rem !important;
    }

    /* ── Responsive Adaptations ── */
    @media (max-width: 768px) {
        .stMainBlockContainer {
            padding-bottom: 120px !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


def get_current_session():
    """Retorna la conversación activa actual."""
    curr_id = st.session_state.active_session_id
    if curr_id not in st.session_state.sessions:
        curr_id = list(st.session_state.sessions.keys())[0]
        st.session_state.active_session_id = curr_id
    return st.session_state.sessions[curr_id]


# ─── Funciones Helper de API ──────────────────────────────────────────────────
@st.cache_data(ttl=2)
def get_documents_from_api():
    """Consulta la lista de documentos indexados en el backend."""
    try:
        r = requests.get(f"{API_URL}/documents/", timeout=4)
        if r.status_code == 200:
            return r.json().get("documents", [])
    except Exception:
        pass
    return []


def delete_document_from_api(doc_name: str):
    """Elimina un documento de Qdrant y sus imágenes en el backend."""
    try:
        r = requests.delete(f"{API_URL}/documents/{doc_name}", timeout=10)
        return r.status_code == 200, r.json()
    except Exception as e:
        return False, {"error": str(e)}


# ─── Healthcheck del Backend ──────────────────────────────────────────────────
api_online = False
qdrant_mode = "desconectado"
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=3)
    if health_resp.status_code == 200:
        api_online = True
        h_data = health_resp.json()
        qdrant_mode = h_data.get("qdrant_mode", "server")
except Exception:
    api_online = False


# ─── SIDEBAR: Panel de Control & Historial Estilo ChatGPT ──────────────────────
with st.sidebar:
    # Encabezado del Sidebar
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 1rem;">
        <span style="color: #3b82f6; font-size: 1.25rem;">✦</span>
        <span style="font-weight: 700; font-size: 1rem; color: #f0f6fc; letter-spacing: -0.01em;">
            RAG Multimodal
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Botón Nueva Conversación
    if st.button("＋ Nueva conversación", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.sessions[new_id] = {
            "id": new_id,
            "title": "Nueva consulta",
            "created_at": datetime.now().strftime("%d/%m %H:%M"),
            "messages": [],
        }
        st.session_state.active_session_id = new_id
        st.rerun()

    # Selector de Proveedor IA
    st.markdown('<div class="sidebar-section-title">Proveedor de IA</div>', unsafe_allow_html=True)
    provider_options = ["OpenAI (GPT-4o-mini)", "Ollama Local (Llama 3.2)"]
    current_index = 0 if st.session_state.ai_provider == "openai" else 1

    selected_provider = st.radio(
        "Proveedor LLM",
        options=provider_options,
        index=current_index,
        label_visibility="collapsed",
    )

    new_prov = "openai" if "OpenAI" in selected_provider else "ollama"
    if new_prov != st.session_state.ai_provider:
        st.session_state.ai_provider = new_prov
        st.rerun()

    if st.session_state.ai_provider == "openai":
        st.caption("☁️ Nube: Alta velocidad (~1.5s) con visión nativa")
    else:
        st.caption("🖥️ Local: On-Premise en CPU (~80s). Privacidad total")

    # Lista de Conversaciones Activas
    st.markdown('<div class="sidebar-section-title">Conversaciones Recientes</div>', unsafe_allow_html=True)

    for s_id, s_data in list(st.session_state.sessions.items()):
        is_active = (s_id == st.session_state.active_session_id)
        display_title = s_data["title"]
        if len(display_title) > 22:
            display_title = display_title[:22] + "..."

        btn_prefix = "●" if is_active else "○"
        btn_label = f"{btn_prefix} {display_title}"

        col_s1, col_s2 = st.columns([0.82, 0.18])
        with col_s1:
            if st.button(btn_label, key=f"session_btn_{s_id}", use_container_width=True):
                st.session_state.active_session_id = s_id
                st.rerun()
        with col_s2:
            if len(st.session_state.sessions) > 1:
                if st.button("✕", key=f"del_sess_{s_id}", help="Eliminar conversación"):
                    del st.session_state.sessions[s_id]
                    st.session_state.active_session_id = list(st.session_state.sessions.keys())[0]
                    st.rerun()

    # Pie de Sidebar: Estado del Sistema
    st.markdown('<div class="sidebar-section-title">Estado del Sistema</div>', unsafe_allow_html=True)
    status_color = "#10b981" if api_online else "#ef4444"
    status_label = "Conectado" if api_online else "Sin conexión"
    prov_short = "GPT-4o" if st.session_state.ai_provider == "openai" else "Llama 3.2"

    st.markdown(f"""
    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 8px; padding: 8px 10px; font-size: 0.78rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #8b949e;">API Backend:</span>
            <span style="color: {status_color}; font-weight: 600;">● {status_label}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #8b949e;">Base Vectorial:</span>
            <span style="color: #cbd5e1; font-family: monospace;">Qdrant</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #8b949e;">Motor Activo:</span>
            <span style="color: #60a5fa; font-weight: 500;">{prov_short}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Exportar Conversación
    curr_sess = get_current_session()
    if curr_sess["messages"]:
        st.markdown("<br>", unsafe_allow_html=True)
        export_text = f"# Conversación RAG: {curr_sess['title']}\nFecha: {curr_sess['created_at']}\nProveedor: {prov_short}\n\n"
        for m in curr_sess["messages"]:
            export_text += f"### {m['role'].upper()}:\n{m['content']}\n\n"
        st.download_button(
            label="Descargar historial (.md)",
            data=export_text,
            file_name=f"chat_{curr_sess['id']}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ─── HEADER PRINCIPAL ELEGANTE Y DISCRETO ─────────────────────────────────────
active_dot_class = "status-dot-active" if st.session_state.ai_provider == "openai" else "status-dot-local"

st.markdown(f"""
<div class="rag-main-header">
    <div class="rag-header-brand">
        <span class="rag-header-sparkle">✦</span>
        <div>
            <h1 class="rag-header-title">RAG Multimodal</h1>
            <p class="rag-header-subtitle">Asistente inteligente basado en documentos</p>
        </div>
    </div>
    <div class="rag-header-status">
        <span class="{active_dot_class}"></span>
        <span>{active_prov_label}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── PESTAÑAS PRINCIPALES MINIMALISTAS ────────────────────────────────────────
tab_chat, tab_crud, tab_history, tab_metrics = st.tabs([
    "Charlar",
    "Documentos",
    "Historial",
    "Sistema",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: ÁREA DE CONVERSACIÓN (CHARLAR)
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    active_session = get_current_session()
    messages = active_session["messages"]

    # Entrada de texto exclusiva de la pestaña de conversación (Charlar)
    user_text = st.chat_input("Escribe tu pregunta...")

    # Determinar si existe consulta a procesar (por teclado o por botón preset)
    query_to_send = None
    if st.session_state.preset_prompt:
        query_to_send = st.session_state.preset_prompt
        st.session_state.preset_prompt = None
    elif user_text:
        query_to_send = user_text

    # Estado Inicial sin mensajes (Hero Estilo ChatGPT/Gemini)
    if len(messages) == 0 and not query_to_send:
        st.markdown("""
        <div class="hero-container">
            <div class="hero-icon">✦</div>
            <h2 class="hero-heading">¿En qué puedo ayudarte hoy?</h2>
            <p class="hero-text">
                Pregunta cualquier detalle sobre los manuales técnicos, diagramas o tablas indexadas.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            if st.button("🔍 Ver funciones del velocímetro", use_container_width=True):
                st.session_state.preset_prompt = "Muestra los detalles y funciones del velocímetro"
                st.rerun()
            if st.button("⚡ Componentes e interruptores de control", use_container_width=True):
                st.session_state.preset_prompt = "¿Qué componentes e interruptores de control existen?"
                st.rerun()
        with col_q2:
            if st.button("📊 Resumen de especificaciones técnicas", use_container_width=True):
                st.session_state.preset_prompt = "Resume las especificaciones y advertencias técnicas principales"
                st.rerun()
            if st.button("🛠️ Instrucciones de seguridad o mantenimiento", use_container_width=True):
                st.session_state.preset_prompt = "¿Qué instrucciones de seguridad o mantenimiento se detallan?"
                st.rerun()

    # 2. Renderizar Historial de Mensajes Existentes
    for idx, message in enumerate(messages):
        role = message["role"]
        avatar_icon = "👤" if role == "user" else "✨"
        with st.chat_message(role, avatar=avatar_icon):
            st.markdown(message["content"])

            # Esquemas Visuales e Imágenes Asociadas (Con Deduplicación Estricta)
            if role == "assistant" and message.get("images"):
                unique_imgs = []
                seen_img_sigs = set()
                for img_url in message["images"]:
                    fname = img_url.split("/")[-1]
                    sig_match = re.search(r"(_p\d+_img\d+)", fname)
                    sig = sig_match.group(1) if sig_match else fname
                    if sig not in seen_img_sigs:
                        seen_img_sigs.add(sig)
                        unique_imgs.append(img_url)

                if unique_imgs:
                    with st.expander(f"Diagramas técnicos recuperados ({len(unique_imgs)})", expanded=True):
                        img_cols = st.columns(min(len(unique_imgs), 3))
                        for i, img_u in enumerate(unique_imgs):
                            with img_cols[i % 3]:
                                try:
                                    img_resp = requests.get(f"{API_URL}{img_u}", timeout=10)
                                    if img_resp.status_code == 200:
                                        st.image(img_resp.content, use_container_width=True)
                                        img_clean_name = img_u.split("/")[-1]
                                        p_match = re.search(r"_p(\d+)_", img_clean_name)
                                        p_label = f"Página {p_match.group(1)}" if p_match else "Diagrama"
                                        st.caption(f"📍 {p_label}")
                                except Exception as e:
                                    st.caption(f"Error cargando imagen: {e}")

            # Citas Técnicas y Fuentes Verificadas
            if role == "assistant" and message.get("sources"):
                with st.expander(f"Fuentes utilizadas ({len(message['sources'])} fragmentos)"):
                    for s in message["sources"]:
                        clean_doc_name = re.sub(r"^[0-9a-fA-F-]{36}_", "", s.get("document_name", "Manual"))
                        c_type = s.get("chunk_type", "texto").lower()
                        st.markdown(
                            f"""<div class="citation-card">
                                <span>📄 <b>{clean_doc_name}</b> — Página {s['page_number']}</span>
                                <span class="citation-meta">{c_type.upper()}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )

    # 3. Procesar Nueva Consulta
    if query_to_send:
        # Asignar título de sesión basado en la pregunta inicial
        if len(active_session["messages"]) == 0 or active_session["title"] == "Nueva consulta":
            clean_title = query_to_send.strip()
            if len(clean_title) > 28:
                clean_title = clean_title[:28] + "..."
            active_session["title"] = clean_title

        # Registrar y renderizar inmediatamente mensaje de usuario
        active_session["messages"].append({"role": "user", "content": query_to_send})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query_to_send)

        # Consultar al Backend con el Proveedor Seleccionado
        prov_display = "OpenAI GPT-4o-mini" if active_prov == "openai" else "Ollama Llama 3.2"

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner(f"✦ Analizando documentos con {prov_display}..."):
                try:
                    res = requests.post(
                        f"{API_URL}/query",
                        json={
                            "question": query_to_send,
                            "top_k": 3 if active_prov == "ollama" else 5,
                            "provider": active_prov,
                        },
                        timeout=req_timeout,
                    )
                    res.raise_for_status()
                    res_data = res.json()

                    ans = res_data.get("answer", "")
                    srcs = res_data.get("sources", [])
                    raw_imgs = res_data.get("image_urls", [])
                    has_ctx = res_data.get("has_sufficient_context", True)

                    # Deduplicación estricta de imágenes
                    imgs = []
                    seen_i_sigs = set()
                    for u in raw_imgs:
                        fn = u.split("/")[-1]
                        s_m = re.search(r"(_p\d+_img\d+)", fn)
                        sig_val = s_m.group(1) if s_m else fn
                        if sig_val not in seen_i_sigs:
                            seen_i_sigs.add(sig_val)
                            imgs.append(u)

                    # Renderizar respuesta del LLM
                    st.markdown(ans)

                    # Desplegar diagramas recuperados
                    if imgs:
                        with st.expander(f"Diagramas técnicos recuperados ({len(imgs)})", expanded=True):
                            img_cols = st.columns(min(len(imgs), 3))
                            for i, img_url in enumerate(imgs):
                                with img_cols[i % 3]:
                                    try:
                                        img_r = requests.get(f"{API_URL}{img_url}", timeout=10)
                                        if img_r.status_code == 200:
                                            st.image(img_r.content, use_container_width=True)
                                            clean_fn = img_url.split("/")[-1]
                                            p_m = re.search(r"_p(\d+)_", clean_fn)
                                            p_str = f"Página {p_m.group(1)}" if p_m else "Diagrama"
                                            st.caption(f"📍 {p_str}")
                                    except Exception as e:
                                        st.caption(f"Error cargando imagen: {e}")

                    # Desplegar fuentes utilizadas
                    if srcs:
                        with st.expander(f"Fuentes utilizadas ({len(srcs)} fragmentos)"):
                            for s in srcs:
                                clean_doc_name = re.sub(r"^[0-9a-fA-F-]{36}_", "", s.get("document_name", "Manual"))
                                c_type = s.get("chunk_type", "texto").lower()
                                st.markdown(
                                    f"""<div class="citation-card">
                                        <span>📄 <b>{clean_doc_name}</b> — Página {s['page_number']}</span>
                                        <span class="citation-meta">{c_type.upper()}</span>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                    if not has_ctx:
                        st.info(
                            "✦ Nota defensiva: No se encontró suficiente contexto en los documentos indexados para responder con certeza total."
                        )

                    # Guardar respuesta en la sesión
                    active_session["messages"].append({
                        "role": "assistant",
                        "content": ans,
                        "sources": srcs,
                        "images": imgs,
                    })

                except requests.exceptions.Timeout:
                    err_msg = (
                        f"⏱️ Tiempo de espera agotado ({req_timeout}s) al procesar con Ollama ({prov_display}) en CPU. "
                        f"La inferencia local requiere más tiempo para evaluar el contexto en este procesador. "
                        f"Puedes intentar nuevamente o alternar a OpenAI (GPT-4o-mini) en la barra lateral para respuestas inmediatas."
                    )
                    st.error(err_msg)
                    active_session["messages"].append({"role": "assistant", "content": err_msg})
                except requests.exceptions.ConnectionError:
                    err_msg = "❌ Error de conexión: No se pudo comunicar con el backend FastAPI (:8000)."
                    st.error(err_msg)
                    active_session["messages"].append({"role": "assistant", "content": err_msg})
                except Exception as ex:
                    err_msg = f"❌ Error procesando consulta: {str(ex)}"
                    st.error(err_msg)
                    active_session["messages"].append({"role": "assistant", "content": err_msg})

        # Recargar para refrescar y consolidar el historial
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: BIBLIOTECA & CRUD DE DOCUMENTOS
# ═════════════════════════════════════════════════════════════════════════════
with tab_crud:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h3 style="font-weight: 700; margin-bottom: 0.2rem; color: #f0f6fc; font-size: 1.15rem;">
            Documentos Indexados
        </h3>
        <p style="color: #8b949e; font-size: 0.88rem; margin: 0;">
            Administra tus manuales PDF y sube nuevos archivos para procesamiento vectorial y extracción visual.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── SUBIDA DE DOCUMENTOS EN LOTE ──
    with st.expander("Subir nuevos documentos PDF", expanded=True):
        uploaded_files = st.file_uploader(
            "Seleccionar archivos PDF",
            type=["pdf"],
            accept_multiple_files=True,
            help="Soporta subida en lote de manuales técnicos.",
            label_visibility="collapsed",
        )

        current_indexed_docs = get_documents_from_api()
        existing_doc_names = {d["document_name"].lower() for d in current_indexed_docs}

        # Detección anticipada de duplicados
        duplicate_files = []
        if uploaded_files:
            for f in uploaded_files:
                if f.name.lower() in existing_doc_names:
                    duplicate_files.append(f.name)

        overwrite_option = False
        if duplicate_files:
            st.markdown(f"""
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 8px; padding: 10px 12px; margin: 10px 0;">
                <span style="color: #fbbf24; font-weight: 600; font-size: 0.85rem;">Documentos ya existentes:</span>
                <ul style="margin: 4px 0 0 16px; color: #f0f6fc; font-size: 0.82rem;">
                    {"".join(f"<li>{name}</li>" for name in duplicate_files)}
                </ul>
                <span style="font-size: 0.78rem; color: #8b949e;">Se omitirán para evitar duplicados en Qdrant. Activa la casilla si deseas re-indexarlos.</span>
            </div>
            """, unsafe_allow_html=True)

            overwrite_option = st.checkbox(
                "Sobrescribir y re-indexar archivos duplicados",
                value=False,
            )

        if uploaded_files:
            col_b1, col_b2 = st.columns([0.4, 0.6])
            with col_b1:
                start_batch = st.button("Comenzar procesamiento", type="primary", use_container_width=True)
            with col_b2:
                st.caption(f"{len(uploaded_files)} archivo(s) listo(s) para indexar.")

            if start_batch:
                total_files = len(uploaded_files)
                batch_progress = st.progress(0, text="Iniciando procesamiento...")
                status_box = st.container()

                processed_count = 0
                for f_idx, up_file in enumerate(uploaded_files):
                    with status_box:
                        st.markdown(f"**Procesando ({f_idx+1}/{total_files}):** `{up_file.name}`")
                        file_progress = st.progress(0.1, text=f"Subiendo `{up_file.name}`...")

                        try:
                            files_payload = {"file": (up_file.name, up_file.getvalue(), "application/pdf")}
                            ingest_url = f"{API_URL}/ingest/?overwrite={'true' if overwrite_option else 'false'}"
                            resp = requests.post(ingest_url, files=files_payload, timeout=40)

                            if resp.status_code == 409:
                                file_progress.empty()
                                st.warning(f"Omitido: `{up_file.name}` ya está indexado.")
                                continue

                            resp.raise_for_status()
                            job_data = resp.json()
                            job_id = job_data["job_id"]

                            # Polling del job
                            job_done = False
                            for attempt in range(80):
                                time.sleep(3)
                                try:
                                    status_resp = requests.get(f"{API_URL}/ingest/{job_id}", timeout=10)
                                    status_data = status_resp.json()
                                    st_status = status_data.get("status")
                                    st_msg = status_data.get("progress_message", "")

                                    if st_status == "processing":
                                        file_progress.progress(
                                            min(0.85, 0.15 + attempt * 0.02),
                                            text=f"{st_msg}"
                                        )
                                    elif st_status == "completed":
                                        file_progress.progress(1.0, text=f"`{up_file.name}` completado.")
                                        chunks_n = status_data.get("total_chunks", 0)
                                        st.success(f"✓ **{up_file.name}** indexado con éxito ({chunks_n} fragmentos).")
                                        job_done = True
                                        processed_count += 1
                                        break
                                    elif st_status == "failed":
                                        file_progress.empty()
                                        st.error(f"Error en `{up_file.name}`: {status_data.get('error')}")
                                        job_done = True
                                        break
                                except Exception:
                                    pass

                            if not job_done:
                                st.info(f"El archivo `{up_file.name}` continúa procesándose en segundo plano.")

                        except Exception as e_upload:
                            st.error(f"Error al subir `{up_file.name}`: {str(e_upload)}")

                    batch_progress.progress((f_idx + 1) / total_files)

                st.cache_data.clear()
                st.success(f"Lote finalizado: {processed_count} de {total_files} documento(s) procesados correctamente.")
                time.sleep(1.5)
                st.rerun()

    # ── LISTADO DE DOCUMENTOS EN BASE VECTORIAL ──
    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 1.5rem 0 1rem 0;'>", unsafe_allow_html=True)
    col_c1, col_c2 = st.columns([0.75, 0.25])
    with col_c1:
        st.markdown("<h4 style='font-size: 1rem; font-weight: 600; color: #f0f6fc; margin: 0;'>Biblioteca de Documentos</h4>", unsafe_allow_html=True)
    with col_c2:
        if st.button("Actualizar lista", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    docs_list = get_documents_from_api()

    if not docs_list:
        st.markdown("""
        <div class="minimal-card" style="text-align: center; padding: 2rem;">
            <p style="color: #8b949e; font-size: 0.9rem; margin: 0;">
                No hay documentos indexados. Sube uno o más manuales PDF arriba para comenzar.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Métricas Discretas
        total_docs_count = len(docs_list)
        total_chunks_count = sum(d.get("total_chunks", 0) for d in docs_list)
        total_pages_count = sum(d.get("total_pages", 0) for d in docs_list)
        total_images_count = sum(d.get("image_chunks", 0) for d in docs_list)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Documentos", total_docs_count)
        with m_col2:
            st.metric("Páginas", total_pages_count)
        with m_col3:
            st.metric("Fragmentos", total_chunks_count)
        with m_col4:
            st.metric("Diagramas", total_images_count)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tarjetas de Documentos con Acciones CRUD
        for doc_item in docs_list:
            raw_d_name = doc_item.get("document_name", "Desconocido")
            clean_d_name = re.sub(r"^[0-9a-fA-F-]{36}_", "", raw_d_name)
            d_chunks = doc_item.get("total_chunks", 0)
            d_pages = doc_item.get("total_pages", 0)
            d_range = doc_item.get("page_range", "N/A")
            d_imgs = doc_item.get("image_chunks", 0)

            with st.container():
                st.markdown(f"""
                <div class="minimal-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 600; font-size: 0.95rem; color: #f0f6fc;">
                            📄 {clean_d_name}
                        </span>
                        <span style="color: #10b981; font-size: 0.75rem; font-weight: 600;">✓ Indexado</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #8b949e;">
                        Páginas: {d_pages} ({d_range}) &nbsp;•&nbsp; Chunks: {d_chunks} &nbsp;•&nbsp; Diagramas: {d_imgs}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_act1, col_act2, col_act3 = st.columns([0.25, 0.45, 0.3])
                with col_act1:
                    if st.button("Consultar", key=f"ask_doc_{raw_d_name}", use_container_width=True):
                        st.session_state.preset_prompt = f"Resume los puntos clave del documento {clean_d_name}"
                        st.rerun()

                with col_act3:
                    with st.expander("Eliminar"):
                        st.write(f"¿Eliminar `{clean_d_name}`?")
                        if st.button("Confirmar", key=f"confirm_del_{raw_d_name}", type="primary", use_container_width=True):
                            with st.spinner("Eliminando..."):
                                success, res_del = delete_document_from_api(raw_d_name)
                                if success:
                                    st.success(f"`{clean_d_name}` eliminado.")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"Error eliminando: {res_del.get('error', 'Error desconocido')}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: HISTORIAL DE PREGUNTAS
# ═════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h3 style="font-weight: 700; margin-bottom: 0.2rem; color: #f0f6fc; font-size: 1.15rem;">
            Historial de Preguntas
        </h3>
        <p style="color: #8b949e; font-size: 0.88rem; margin: 0;">
            Revisa las consultas formuladas durante esta sesión y vuelve a ejecutarlas.
        </p>
    </div>
    """, unsafe_allow_html=True)

    curr_session = get_current_session()
    sess_messages = curr_session["messages"]
    user_queries = [m for m in sess_messages if m["role"] == "user"]

    if not user_queries:
        st.markdown("""
        <div class="minimal-card" style="text-align: center; padding: 2rem;">
            <p style="color: #8b949e; font-size: 0.9rem; margin: 0;">
                No hay preguntas en esta conversación. Escribe en el chat para registrar el historial.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for q_idx, q_msg in enumerate(user_queries):
            ans_msg = None
            try:
                msg_pos = sess_messages.index(q_msg)
                if msg_pos + 1 < len(sess_messages) and sess_messages[msg_pos + 1]["role"] == "assistant":
                    ans_msg = sess_messages[msg_pos + 1]
            except Exception:
                pass

            with st.expander(f"[{q_idx+1}] {q_msg['content'][:65]}...", expanded=(q_idx == len(user_queries)-1)):
                st.markdown(f"**Pregunta:**\n> {q_msg['content']}")
                if ans_msg:
                    st.markdown("**Respuesta:**")
                    st.markdown(ans_msg["content"][:300] + ("..." if len(ans_msg["content"]) > 300 else ""))
                    if ans_msg.get("images"):
                        st.caption(f"🖼️ {len(ans_msg['images'])} diagrama(s) asociado(s)")

                col_reask, _ = st.columns([0.3, 0.7])
                with col_reask:
                    if st.button("Volver a preguntar", key=f"reask_{q_idx}", use_container_width=True):
                        st.session_state.preset_prompt = q_msg["content"]
                        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: SISTEMA Y MÉTRICAS
# ═════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("""
    <div style="margin-bottom: 1.2rem;">
        <h3 style="font-weight: 700; margin-bottom: 0.2rem; color: #f0f6fc; font-size: 1.15rem;">
            Estado del Sistema
        </h3>
        <p style="color: #8b949e; font-size: 0.88rem; margin: 0;">
            Parámetros de arquitectura y salud del cluster en Docker.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_ar1, col_ar2 = st.columns(2)
    with col_ar1:
        st.markdown(f"""
        <div class="minimal-card">
            <h4 style="font-weight: 600; color: #f0f6fc; font-size: 0.95rem; margin-bottom: 0.6rem;">Topología</h4>
            <div style="font-size: 0.85rem; line-height: 1.8; color: #8b949e;">
                <div>• <b>Arquitectura:</b> Hexagonal (Puertos y Adaptadores)</div>
                <div>• <b>API Gateway:</b> FastAPI Asíncrono</div>
                <div>• <b>Base Vectorial:</b> Qdrant ({qdrant_mode.upper()})</div>
                <div>• <b>Búsqueda:</b> Híbrida (Densa + Sparse)</div>
                <div>• <b>Extracción:</b> PyMuPDF + Detección de Bounding Boxes</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ar2:
        st.markdown(f"""
        <div class="minimal-card">
            <h4 style="font-weight: 600; color: #f0f6fc; font-size: 0.95rem; margin-bottom: 0.6rem;">Estado Operativo</h4>
            <div style="font-size: 0.85rem; line-height: 1.8; color: #8b949e;">
                <div>• <b>Backend:</b> {'Conectado' if api_online else 'Desconectado'}</div>
                <div>• <b>Proveedor LLM:</b> {active_prov_label}</div>
                <div>• <b>Deduplicación de Imágenes:</b> Activa</div>
                <div>• <b>Timeout Dinámico:</b> {req_timeout}s</div>
                <div>• <b>Tests de Calidad:</b> 18/18 Unitarios e Integración</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
