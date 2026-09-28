"""
DESCARGADOR AUTOMÁTICO DE 100 MANUALES Y DOCUMENTOS TÉCNICOS EN PDF
Fuente: Repositorio público oficial 'tpn/pdfs' (GitHub)
- 975 documentos técnicos reales (hardware, redes, arquitecturas, microprocesadores).
- Descarga concurrente con 8 hilos (rápida y ligera: archivos de 100KB a 2.5MB).
"""
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

DEST_FOLDER = "lote_100_manuales"
TARGET_COUNT = 100
MIN_SIZE = 100_000       # Mínimo 100 KB
MAX_SIZE = 2_500_000     # Máximo 2.5 MB (para descarga rápida en ~1 minuto)

def download_single_pdf(item, folder):
    name = item["name"]
    url = item["download_url"]
    file_path = os.path.join(folder, name)
    if os.path.exists(file_path):
        return True, name, "Ya existía"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(resp.content)
            return True, name, f"{len(resp.content)//1024} KB"
        return False, name, f"Status {resp.status_code}"
    except Exception as e:
        return False, name, str(e)

def main():
    print("=" * 65)
    print("  DESCARGADOR DE 100 MANUALES TÉCNICOS (GITHUB TPN/PDFS)")
    print("=" * 65)
    os.makedirs(DEST_FOLDER, exist_ok=True)

    print("\n🔍 Consultando índice de documentos en GitHub API...")
    api_url = "https://api.github.com/repos/tpn/pdfs/contents"
    try:
        r = requests.get(api_url, timeout=15)
        if r.status_code != 200:
            print(f"❌ Error al consultar GitHub API: {r.status_code}")
            return
        items = r.json()
    except Exception as ex:
        print(f"❌ Error de conexión: {ex}")
        return

    # Filtrar PDFs con tamaño óptimo
    candidates = [
        it for it in items
        if it.get("name", "").lower().endswith(".pdf")
        and MIN_SIZE <= it.get("size", 0) <= MAX_SIZE
        and it.get("download_url")
    ][:TARGET_COUNT]

    print(f"✓ Se seleccionaron {len(candidates)} manuales técnicos para descargar.")
    print(f"📁 Carpeta de destino: ./{DEST_FOLDER}/\n")

    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_single_pdf, it, DEST_FOLDER): it for it in candidates}
        for f in as_completed(futures):
            success, fname, note = f.result()
            completed += 1
            status_icon = "✓" if success else "✗"
            short_name = (fname[:50] + "...") if len(fname) > 53 else fname
            print(f"[{completed:02d}/{len(candidates)}] {status_icon} {short_name:<55} ({note})")

    print("\n" + "=" * 65)
    print(f"🎉 ¡Descarga finalizada! {completed} manuales técnicos listos en '{DEST_FOLDER}'.")
    print("Ahora puedes seleccionarlos todos y subirlos en lote a Streamlit.")
    print("=" * 65)

if __name__ == "__main__":
    main()
