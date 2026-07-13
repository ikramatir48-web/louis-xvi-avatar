"""
Pipeline d'ingestion PDF : extraction de texte via PyMuPDF.

Usage :
    python ingest_pdf.py
    (traite automatiquement tous les PDFs présents dans data/raw/pdf/)

Sortie :
    data/transcripts/pdf_<nom_fichier>.txt
    logs/pdf_ingest.log
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF

# ── Configuration ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdf"
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"
LOG_DIR = PROJECT_ROOT / "logs"
PROCESSED_LOG = LOG_DIR / "pdf_processed.json"

MIN_CHARS_PER_PAGE = 20  # seuil pour détecter un PDF scanné (image) sans texte extractible

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pdf_ingest.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_processed() -> dict:
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_processed(processed: dict):
    with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)


def file_signature(filepath: Path) -> str:
    """Signature simple basée sur taille + date de modif, pour détecter si un fichier a changé."""
    stat = filepath.stat()
    return f"{stat.st_size}_{stat.st_mtime}"


def extract_pdf_text(pdf_path: Path) -> dict:
    """
    Extrait le texte d'un PDF page par page.
    Retourne {text, page_count, low_text_pages} - low_text_pages signale les pages
    probablement scannées (image) où PyMuPDF n'a presque rien trouvé -> nécessiteront l'OCR.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    low_text_pages = []

    for page_num, page in enumerate(doc, 1):
        page_text = page.get_text("text")
        if len(page_text.strip()) < MIN_CHARS_PER_PAGE:
            low_text_pages.append(page_num)
        full_text += page_text + "\n"

    page_count = doc.page_count
    doc.close()

    return {
        "text": full_text.strip(),
        "page_count": page_count,
        "low_text_pages": low_text_pages
    }


def ingest_pdfs(skip_existing: bool = True) -> dict:
    """
    Traite tous les PDFs présents dans data/raw/pdf/.
    Signale les PDFs probablement scannés (nécessitant OCR) sans bloquer le reste.
    """
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    processed = load_processed()
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"Aucun PDF trouvé dans {PDF_DIR}. Dépose les PDFs de Chrystelle ici.")
        return {"success": [], "failed": [], "skipped": [], "needs_ocr": []}

    logger.info(f"{len(pdf_files)} PDF(s) trouvé(s)")

    results = {"success": [], "failed": [], "skipped": [], "needs_ocr": []}

    for i, pdf_path in enumerate(pdf_files, 1):
        filename = pdf_path.name
        sig = file_signature(pdf_path)
        logger.info(f"[{i}/{len(pdf_files)}] Traitement : {filename}")

        if skip_existing and processed.get(filename, {}).get("signature") == sig:
            logger.info(f"  -> Déjà traité (inchangé), ignoré")
            results["skipped"].append(filename)
            continue

        try:
            extraction = extract_pdf_text(pdf_path)
        except Exception as e:
            logger.error(f"  -> Échec extraction : {e}")
            results["failed"].append({"file": filename, "reason": str(e)})
            continue

        text = extraction["text"]
        if len(text) < 50:
            logger.warning(f"  -> Texte quasi vide ({len(text)} caractères) - PDF probablement scanné")
            results["needs_ocr"].append(filename)
            continue

        # Avertissement si certaines pages semblent scannées au milieu d'un PDF sinon correct
        if extraction["low_text_pages"]:
            ratio = len(extraction["low_text_pages"]) / extraction["page_count"]
            if ratio > 0.3:
                logger.warning(
                    f"  -> {len(extraction['low_text_pages'])}/{extraction['page_count']} pages "
                    f"avec peu de texte (possibles pages scannées) : {extraction['low_text_pages'][:10]}"
                )

        # Sauvegarde avec métadonnées en en-tête (même format que YouTube, pour cohérence)
        safe_name = filename.replace(".pdf", "").replace(" ", "_")
        transcript_path = TRANSCRIPT_DIR / f"pdf_{safe_name}.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(f"[SOURCE: PDF]\n")
            f.write(f"[FICHIER: {filename}]\n")
            f.write(f"[PAGES: {extraction['page_count']}]\n")
            f.write("---\n")
            f.write(text)

        logger.info(f"  -> Extrait ({len(text)} caractères, {extraction['page_count']} pages) -> {transcript_path.name}")

        processed[filename] = {
            "signature": sig,
            "transcript_path": str(transcript_path),
            "processed_at": datetime.now().isoformat(),
            "char_count": len(text),
            "page_count": extraction["page_count"]
        }
        results["success"].append(filename)

    save_processed(processed)

    logger.info("─" * 60)
    logger.info(f"Terminé : {len(results['success'])} succès, {len(results['failed'])} échecs, "
                f"{len(results['skipped'])} ignorés, {len(results['needs_ocr'])} nécessitent un OCR")

    if results["needs_ocr"]:
        logger.warning("PDFs scannés détectés (texte non extractible directement) :")
        for f in results["needs_ocr"]:
            logger.warning(f"  - {f} -> nécessite OCR (voir ingest_pdf_ocr.py)")

    return results


if __name__ == "__main__":
    ingest_pdfs()
