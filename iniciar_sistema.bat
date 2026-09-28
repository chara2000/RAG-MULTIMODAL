@echo off
title Sistema RAG Multimodal
chcp 65001 >nul
echo =====================================================================
echo           SISTEMA RAG MULTIMODAL - INICIO RAPIDO
echo =====================================================================
echo.
echo [1/2] Iniciando Servidor Backend (FastAPI en http://localhost:8000)...
start "RAG - Backend (FastAPI)" cmd /k "cd backend && python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload"

echo Esperando 3 segundos a que el backend se inicialice...
timeout /t 3 /nobreak >nul

echo.
echo [2/2] Iniciando Servidor Frontend (Streamlit en http://localhost:8501)...
start "RAG - Frontend (Streamlit)" cmd /k "cd frontend && python -m streamlit run streamlit_app.py --server.port 8501"

echo.
echo =====================================================================
echo   ✅ SERVICIOS EN EJECUCION:
echo   • Frontend (Chat UI):  http://localhost:8501
echo   • Backend (Swagger):   http://localhost:8000/docs
echo   • Health Check:        http://localhost:8000/health
echo =====================================================================
echo.
echo Puedes cerrar esta ventana. Las ventanas de Backend y Frontend
echo continuaran ejecutandose en segundo plano.
echo.
pause
