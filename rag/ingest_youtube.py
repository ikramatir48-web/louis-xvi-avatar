"""
Pipeline d'ingestion YouTube : récupère le texte d'une vidéo en suivant cette priorité :

  1. Sous-titres FR déjà présents sur YouTube (rapide, gratuit, pas d'erreurs de reconnaissance vocale)
  2. Sous-titres auto-générés par YouTube si pas de FR manuel disponible
  3. Whisper en dernier recours (téléchargement audio + transcription, plus lent)

Usage :
    python ingest_youtube.py urls.txt

où urls.txt contient une URL YouTube par ligne (lignes vides et commentaires # ignorés).

Sortie :
    data/transcripts/youtube_<id_video>.txt   (texte, peu importe la méthode utilisée)
    logs/youtube_ingest.log                   (journal des succès/échecs)
    logs/youtube_processed.json               (cache pour ne pas retraiter 2x)
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Force l'UTF-8 pour toutes les opérations I/O de ce script (stdout, fichiers...).
# Sans ça, sur certaines configurations Windows, Python utilise l'encodage régional
# (souvent cp1252) comme encodage par défaut, ce qui corrompt les caractères accentués
# même quand encoding="utf-8" est passé explicitement à open() dans certains cas.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import yt_dlp

# ── Configuration ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "youtube"
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"
LOG_DIR = PROJECT_ROOT / "logs"
PROCESSED_LOG = LOG_DIR / "youtube_processed.json"

# Whisper n'est importé que si besoin (évite de charger le modèle si les sous-titres suffisent)
WHISPER_MODEL_SIZE = "small"  # tiny/base/small/medium - "small" = bon compromis qualité/vitesse en CPU
WHISPER_DURATION_THRESHOLD = 1800  # 30 min : au-delà, on avertit que Whisper sera lent sur CPU

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "youtube_ingest.log", encoding="utf-8"),
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


def read_urls(filepath: str) -> list[str]:
    """
    Lit un fichier texte contenant une URL par ligne.
    Utilise encoding="utf-8-sig" pour gérer automatiquement le BOM (Byte Order Mark)
    que certains éditeurs/terminaux (notamment PowerShell avec `echo > fichier.txt`,
    qui écrit par défaut en UTF-16) peuvent ajouter en tête de fichier.
    """
    urls = []
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
    except UnicodeDecodeError:
        # Fallback : le fichier est probablement en UTF-16 (cas PowerShell "echo >")
        with open(filepath, "r", encoding="utf-16") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
    return urls


def get_video_info(youtube_url: str) -> dict | None:
    """Récupère les métadonnées d'une vidéo sans la télécharger (rapide)."""
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info
    except Exception as e:
        logger.error(f"Échec récupération info {youtube_url} : {e}")
        return None


def vtt_to_plain_text(vtt_content: str) -> str:
    """
    Convertit un fichier de sous-titres .vtt en texte brut continu :
    supprime les timestamps, numéros de séquence, balises, et dédoublonne
    les lignes répétées consécutives (artefact fréquent des sous-titres auto YouTube
    où chaque ligne apparaît 2 fois à cause du défilement progressif).
    """
    lines = vtt_content.split("\n")
    text_lines = []
    seen_recent = []  # fenêtre glissante pour détecter les doublons immédiats

    for line in lines:
        line = line.strip()
        # Ignore : en-tête WEBVTT, timestamps, lignes vides, numéros de séquence purs
        if (not line or line == "WEBVTT" or "-->" in line or line.isdigit()
                or line.startswith("Kind:") or line.startswith("Language:")):
            continue
        # Supprime les balises de positionnement/style type <c> ou <00:00:01.000>
        clean_line = re.sub(r"<[^>]+>", "", line).strip()
        if not clean_line:
            continue
        # Évite les doublons consécutifs (les sous-titres auto YouTube répètent souvent la ligne précédente)
        if clean_line not in seen_recent:
            text_lines.append(clean_line)
        seen_recent.append(clean_line)
        if len(seen_recent) > 3:
            seen_recent.pop(0)

    return " ".join(text_lines)


def try_fetch_subtitles(youtube_url: str, video_id: str) -> str | None:
    """
    Tente de récupérer les sous-titres FR (manuels en priorité, puis auto-générés).
    Retourne le texte brut, ou None si aucun sous-titre n'est disponible.
    """
    sub_dir = RAW_DIR / "subs_tmp"
    sub_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["fr"],
        "subtitlesformat": "vtt",
        "outtmpl": str(sub_dir / f"{video_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except Exception as e:
        logger.warning(f"  -> Pas de sous-titres récupérables : {e}")
        return None

    # yt-dlp nomme le fichier <id>.fr.vtt (manuel) ou <id>.fr.vtt (auto, même extension)
    vtt_candidates = list(sub_dir.glob(f"{video_id}.fr*.vtt"))
    if not vtt_candidates:
        return None

    vtt_path = vtt_candidates[0]
    vtt_content = vtt_path.read_text(encoding="utf-8")
    text = vtt_to_plain_text(vtt_content)

    # Nettoyage du fichier temporaire
    vtt_path.unlink(missing_ok=True)

    return text if len(text.strip()) > 20 else None


def download_audio(youtube_url: str, video_id: str) -> str | None:
    """Télécharge uniquement l'audio, pour le fallback Whisper."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(RAW_DIR / "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return str(RAW_DIR / f"{video_id}.mp3")
    except Exception as e:
        logger.error(f"Échec téléchargement audio {youtube_url} : {e}")
        return None


def transcribe_with_whisper(audio_path: str) -> str | None:
    """Fallback : transcrit l'audio avec Whisper. N'importe whisper qu'à ce moment-là."""
    import whisper
    logger.info(f"  -> Chargement du modèle Whisper '{WHISPER_MODEL_SIZE}' (1-2 min la 1ère fois)...")
    model = whisper.load_model(WHISPER_MODEL_SIZE)
    try:
        result = model.transcribe(audio_path, language="fr", verbose=False)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Échec transcription Whisper {audio_path} : {e}")
        return None


def ingest_youtube_urls(urls: list[str], skip_existing: bool = True) -> dict:
    """
    Pipeline complet par URL :
      1. Récupère les infos vidéo (titre, durée)
      2. Tente les sous-titres FR (manuel puis auto)
      3. Si échec, fallback Whisper (télécharge l'audio + transcrit)
      4. Sauvegarde avec métadonnées + méthode utilisée
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    processed = load_processed()
    results = {"success": [], "failed": [], "skipped": []}

    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{len(urls)}] Traitement : {url}")

        if skip_existing and url in processed:
            logger.info(f"  -> Déjà traité, ignoré (transcript: {processed[url]['transcript_path']})")
            results["skipped"].append(url)
            continue

        info = get_video_info(url)
        if info is None:
            results["failed"].append({"url": url, "reason": "métadonnées introuvables"})
            continue

        video_id = info["id"]
        title = info.get("title", "Sans titre")
        duration = info.get("duration", 0)
        logger.info(f"  -> '{title}' ({duration}s)")

        method_used = None
        text = None

        # 1. Tentative sous-titres
        logger.info("  -> Recherche de sous-titres FR...")
        text = try_fetch_subtitles(url, video_id)
        if text:
            method_used = "sous-titres YouTube"
            logger.info(f"  -> Sous-titres trouvés ({len(text)} caractères)")
        else:
            # 2. Fallback Whisper
            logger.info("  -> Pas de sous-titres exploitables, fallback Whisper")
            if duration > WHISPER_DURATION_THRESHOLD:
                logger.warning(f"  -> Vidéo longue ({duration//60} min) : la transcription Whisper "
                               f"sur CPU peut prendre {duration // 60 // 2}-{duration // 60} minutes")
            audio_path = download_audio(url, video_id)
            if audio_path is None:
                results["failed"].append({"url": url, "reason": "téléchargement audio échoué"})
                continue
            text = transcribe_with_whisper(audio_path)
            method_used = "whisper"

        if text is None or len(text.strip()) < 20:
            results["failed"].append({"url": url, "reason": "aucun texte exploitable obtenu"})
            continue

        # Sauvegarde avec métadonnées (la méthode utilisée est tracée pour info/debug)
        transcript_path = TRANSCRIPT_DIR / f"youtube_{video_id}.txt"
        file_content = (
            f"[SOURCE: YouTube]\n"
            f"[TITRE: {title}]\n"
            f"[URL: {url}]\n"
            f"[DUREE: {duration}s]\n"
            f"[METHODE: {method_used}]\n"
            f"---\n"
            f"{text}"
        )
        # Écriture en bytes UTF-8 explicite (voir commentaire en tête de fichier)
        with open(transcript_path, "wb") as f:
            f.write(file_content.encode("utf-8"))

        logger.info(f"  -> Transcrit via {method_used} ({len(text)} caractères) -> {transcript_path.name}")

        processed[url] = {
            "video_id": video_id,
            "title": title,
            "method": method_used,
            "transcript_path": str(transcript_path),
            "processed_at": datetime.now().isoformat(),
            "char_count": len(text)
        }
        results["success"].append(url)

    save_processed(processed)

    logger.info("─" * 60)
    logger.info(f"Terminé : {len(results['success'])} succès, "
                f"{len(results['failed'])} échecs, {len(results['skipped'])} ignorés")
    if results["failed"]:
        logger.warning("Échecs détaillés :")
        for fail in results["failed"]:
            logger.warning(f"  - {fail['url']} : {fail['reason']}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python ingest_youtube.py urls.txt")
        sys.exit(1)

    urls_file = sys.argv[1]
    urls = read_urls(urls_file)
    logger.info(f"{len(urls)} URL(s) trouvée(s) dans {urls_file}")

    ingest_youtube_urls(urls)