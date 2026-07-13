"""
Pipeline de nettoyage : transforme les transcripts bruts (data/transcripts/)
en texte propre et normalisé (data/cleaned/), prêt à être découpé en chunks.

Usage :
    python clean_text.py

Sortie :
    data/cleaned/<meme_nom>.txt
    logs/cleaning_report.json   (statistiques avant/après par fichier)
"""

import re
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Force l'UTF-8 pour toutes les opérations I/O de ce script (voir ingest_youtube.py
# pour le détail du problème que ça résout sous Windows).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ── Configuration ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
LOG_DIR = PROJECT_ROOT / "logs"
REPORT_PATH = LOG_DIR / "cleaning_report.json"

MIN_CLEAN_LENGTH = 100  # en dessous, on signale le fichier comme suspect

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "cleaning.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def extract_metadata_header(raw_text: str) -> tuple[dict, str]:
    """
    Sépare l'en-tête de métadonnées [SOURCE: ...] [TITRE: ...] etc. du contenu réel.
    Retourne (metadata_dict, contenu_sans_entete).
    """
    metadata = {}
    lines = raw_text.split("\n")
    content_start = 0

    for i, line in enumerate(lines):
        match = re.match(r"\[(\w+):\s*(.+?)\]", line.strip())
        if match:
            metadata[match.group(1).lower()] = match.group(2)
        elif line.strip() == "---":
            content_start = i + 1
            break
        elif i > 10:
            # Pas d'en-tête détecté dans les 10 premières lignes -> pas de métadonnées
            content_start = 0
            break

    content = "\n".join(lines[content_start:])
    return metadata, content


def clean_youtube_transcript(text: str) -> str:
    """
    Nettoyage spécifique aux transcriptions Whisper :
    - timestamps résiduels type [00:12:34]
    - répétitions immédiates de mots/phrases (artefact fréquent de Whisper)
    - hésitations orales (euh, hum...) en excès
    - espaces multiples
    """
    # Timestamps
    text = re.sub(r"\[\d{1,2}:\d{2}(:\d{2})?\]", "", text)

    # Répétitions immédiates de mots (ex: "le le chat" -> "le chat")
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text, flags=re.IGNORECASE)

    # Hésitations orales isolées (avec la ponctuation/espace qui traîne juste avant, ex: ", euh,")
    text = re.sub(r"\s*,?\s*\b(euh+|hum+|heu+)\b\s*,?", " ", text, flags=re.IGNORECASE)

    return text


def clean_pdf_text(text: str) -> str:
    """
    Nettoyage spécifique à l'extraction PDF :
    - sauts de ligne intempestifs au milieu de phrases (artefact de mise en page PDF)
    - numéros de page isolés
    - en-têtes/pieds de page répétés (heuristique simple : lignes courtes très répétées)
    """
    # Numéros de page isolés sur leur propre ligne
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Recolle les lignes coupées en plein milieu de phrase
    # (ligne qui ne se termine pas par ponctuation forte, suivie d'une ligne en minuscule)
    text = re.sub(r"([a-zàâäéèêëïîôöùûüç,])\n([a-zàâäéèêëïîôöùûüç])", r"\1 \2", text)

    # Tirets de coupure de mots en fin de ligne (césure typographique : "histo-\nrique" -> "historique")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text


def clean_common(text: str) -> str:
    """Nettoyage générique appliqué à toutes les sources, après le nettoyage spécifique."""
    # Normalise les espaces multiples
    text = re.sub(r"[ \t]+", " ", text)

    # Normalise les sauts de ligne multiples (max 2 consécutifs = un paragraphe)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Supprime les espaces en début/fin de ligne
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Supprime les lignes totalement vides en trop
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()


def remove_repeated_boilerplate(texts: list[str], threshold: float = 0.6) -> list[str]:
    """
    Détecte les lignes courtes répétées dans un grand nombre de documents
    (typiquement en-têtes/pieds de page de livres scannés type "Histoire de France - Tome II")
    et les retire. threshold = fraction de documents dans lesquels la ligne doit apparaître.
    """
    from collections import Counter

    line_counter = Counter()
    doc_count = len(texts)

    for text in texts:
        lines = set(l.strip() for l in text.split("\n") if 0 < len(l.strip()) < 80)
        for line in lines:
            line_counter[line] += 1

    boilerplate = {line for line, count in line_counter.items()
                   if count / doc_count >= threshold and doc_count > 1}

    if boilerplate:
        logger.info(f"  -> {len(boilerplate)} ligne(s) répétitive(s) détectée(s) et retirée(s) "
                    f"(ex: {list(boilerplate)[:3]})")

    cleaned_texts = []
    for text in texts:
        lines = [l for l in text.split("\n") if l.strip() not in boilerplate]
        cleaned_texts.append("\n".join(lines))

    return cleaned_texts


def clean_file(filepath: Path) -> dict:
    """Nettoie un fichier transcript individuel. Retourne un rapport avant/après."""
    raw_text = filepath.read_text(encoding="utf-8")
    metadata, content = extract_metadata_header(raw_text)

    source_type = metadata.get("source", "").lower()

    if source_type == "youtube":
        content = clean_youtube_transcript(content)
    elif source_type == "pdf":
        content = clean_pdf_text(content)

    content = clean_common(content)

    return {
        "metadata": metadata,
        "content": content,
        "original_length": len(raw_text),
        "cleaned_length": len(content)
    }


def run_cleaning_pipeline():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.txt"))
    if not transcript_files:
        logger.warning(f"Aucun transcript trouvé dans {TRANSCRIPT_DIR}. "
                       f"Lance d'abord ingest_youtube.py et/ou ingest_pdf.py")
        return

    logger.info(f"{len(transcript_files)} fichier(s) à nettoyer")

    # Passe 1 : nettoyage individuel
    cleaned_results = {}
    for filepath in transcript_files:
        logger.info(f"Nettoyage : {filepath.name}")
        result = clean_file(filepath)
        cleaned_results[filepath.name] = result

    # Passe 2 : détection du boilerplate répété à travers tous les documents
    # (utile pour les livres/PDF qui partagent un en-tête de collection par ex.)
    all_contents = [r["content"] for r in cleaned_results.values()]
    if len(all_contents) > 1:
        deduped_contents = remove_repeated_boilerplate(all_contents)
        for (filename, result), new_content in zip(cleaned_results.items(), deduped_contents):
            result["content"] = clean_common(new_content)
            result["cleaned_length"] = len(result["content"])

    # Sauvegarde + rapport
    report = {}
    suspects = []

    for filename, result in cleaned_results.items():
        out_path = CLEANED_DIR / filename
        # Écriture en bytes UTF-8 explicite (voir commentaire en tête de fichier)
        with open(out_path, "wb") as f:
            f.write(result["content"].encode("utf-8"))

        reduction_pct = round(
            (1 - result["cleaned_length"] / max(result["original_length"], 1)) * 100, 1
        )
        report[filename] = {
            "original_length": result["original_length"],
            "cleaned_length": result["cleaned_length"],
            "reduction_pct": reduction_pct,
            "metadata": result["metadata"]
        }

        if result["cleaned_length"] < MIN_CLEAN_LENGTH:
            suspects.append(filename)

        logger.info(f"  {filename} : {result['original_length']} -> {result['cleaned_length']} "
                    f"caractères (-{reduction_pct}%)")

    report["_generated_at"] = datetime.now().isoformat()
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("─" * 60)
    logger.info(f"Nettoyage terminé : {len(cleaned_results)} fichiers traités")
    logger.info(f"Rapport détaillé : {REPORT_PATH}")

    if suspects:
        logger.warning(f"⚠ {len(suspects)} fichier(s) suspect(s) (très court après nettoyage, "
                       f"à vérifier manuellement) : {suspects}")


if __name__ == "__main__":
    run_cleaning_pipeline()