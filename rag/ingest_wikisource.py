"""
Pipeline d'ingestion Wikisource : récupère le texte d'une page Wikisource via son API,
beaucoup plus fiable que le scraping HTML classique (pas de pub, pas de menu, juste le contenu).

Usage :
    python ingest_wikisource.py wikisource_urls.txt

où wikisource_urls.txt contient une URL Wikisource par ligne.

Sortie :
    data/transcripts/wikisource_<titre>.txt
    logs/wikisource_ingest.log
"""

import sys
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import unquote, urlparse

# Force l'UTF-8 pour toutes les opérations I/O de ce script (voir ingest_youtube.py
# pour le détail du problème que ça résout sous Windows).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import requests

# ── Configuration ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"
LOG_DIR = PROJECT_ROOT / "logs"
PROCESSED_LOG = LOG_DIR / "wikisource_processed.json"

# ── Logging ────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "wikisource_ingest.log", encoding="utf-8"),
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


def extract_page_title(url: str) -> str:
    """Extrait le titre de la page depuis l'URL Wikisource (ex: 'Testament_de_Louis_XVI')."""
    path = urlparse(url).path
    title = path.split("/wiki/")[-1]
    return unquote(title)


def fetch_wikisource_text(url: str) -> dict | None:
    """
    Récupère le texte d'une page Wikisource via le rendu HTML de l'API MediaWiki (action=parse).
    Contrairement à prop=extracts, cette méthode suit correctement les pages de type
    "Livre" (transclusion de sous-pages), très courantes pour les documents historiques
    sur Wikisource (le texte y est stocké page par page puis assemblé).
    """
    title = extract_page_title(url)
    api_url = "https://fr.wikisource.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
        "redirects": True   # suit automatiquement les redirections (ex: apostrophes typographiques différentes)
    }
    headers = {"User-Agent": "ResearchBot/1.0 (projet interview historique educatif)"}

    try:
        response = requests.get(api_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()

        # Important : on ignore le charset deviné par requests/le serveur et on décode
        # nous-mêmes les bytes bruts en UTF-8. L'API MediaWiki renvoie du JSON en UTF-8,
        # mais requests peut mal détecter ce charset depuis l'en-tête Content-Type et
        # appliquer un mauvais décodage AVANT le json.loads(), ce qui corrompt définitivement
        # les caractères accentués (le mojibake est alors déjà dans la chaîne Python, écrire
        # en bytes ensuite n'y change rien).
        import json as json_module
        data = json_module.loads(response.content.decode("utf-8"))

        if "error" in data:
            logger.warning(f"  -> Page introuvable sur Wikisource : {title} ({data['error'].get('info', '')})")
            return None

        html = data.get("parse", {}).get("text", "")
        page_title = data.get("parse", {}).get("title", title)

        if not html:
            logger.warning(f"  -> Pas de contenu HTML pour : {title}")
            return None

        text = html_to_plain_text(html)

        if not text or len(text.strip()) < 30:
            logger.warning(f"  -> Contenu vide ou trop court pour : {title}")
            return None

        return {"title": page_title, "text": text}
    except Exception as e:
        logger.error(f"  -> Erreur API Wikisource pour {url} : {e}")
        return None


def html_to_plain_text(html: str) -> str:
    """
    Convertit le HTML rendu par MediaWiki en texte brut.
    Retire : scripts/styles, références [1], boîtes de navigation, tableaux de métadonnées,
    et les balises de structure interne à Wikisource (class="reference", "noprint", etc.)
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Supprime les éléments non-textuels ou parasites
    for tag in soup.find_all(["script", "style", "sup", "table"]):
        tag.decompose()
    for tag in soup.find_all(class_=["reference", "noprint", "mw-editsection", "infobox_v2"]):
        tag.decompose()

    # Insère un espace avant chaque balise <span> et <a> : sans ça, des éléments
    # adjacents sans saut de ligne (ex: <span>nale</span><span>signifiés</span>)
    # fusionnent en un seul mot "nalesignifiés" lors de l'extraction du texte.
    for tag in soup.find_all(["span", "a"]):
        tag.insert_before(" ")

    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]{2,}", " ", text)  # nettoie les espaces doublés introduits ci-dessus
    return text


def clean_wikisource_text(text: str) -> str:
    """
    Nettoyage léger spécifique à Wikisource :
    - retire les marqueurs de note de bas de page type [1], [a]
    - retire les lignes de métadonnées résiduelles (rare avec l'API extracts mais possible)
    """
    text = re.sub(r"\[\d+\]", "", text)          # [1], [23]...
    text = re.sub(r"\[[a-z]\]", "", text)          # [a], [b]...
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ingest_wikisource_urls(urls: list[str], skip_existing: bool = True) -> dict:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    processed = load_processed()
    results = {"success": [], "failed": [], "skipped": []}

    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{len(urls)}] Traitement : {url}")

        if skip_existing and url in processed:
            logger.info(f"  -> Déjà traité, ignoré")
            results["skipped"].append(url)
            continue

        result = fetch_wikisource_text(url)
        if result is None:
            results["failed"].append({"url": url, "reason": "récupération échouée ou page vide"})
            continue

        title = result["title"]
        text = clean_wikisource_text(result["text"])

        safe_name = re.sub(r"[^\w\-]", "_", title)[:80]
        transcript_path = TRANSCRIPT_DIR / f"wikisource_{safe_name}.txt"

        file_content = (
            f"[SOURCE: Wikisource]\n"
            f"[TITRE: {title}]\n"
            f"[URL: {url}]\n"
            f"---\n"
            f"{text}"
        )
        # Écriture explicite en bytes UTF-8 (et non via le mode texte "w") :
        # sur certaines configurations Windows, Python utilise l'encodage de la
        # console (souvent cp1252) comme encodage par défaut au niveau système,
        # ce qui peut provoquer un double encodage même avec encoding="utf-8"
        # passé à open(). Écrire en mode binaire élimine toute ambiguïté.
        with open(transcript_path, "wb") as f:
            f.write(file_content.encode("utf-8"))

        logger.info(f"  -> Récupéré : '{title}' ({len(text)} caractères) -> {transcript_path.name}")

        processed[url] = {
            "title": title,
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
        for fail in results["failed"]:
            logger.warning(f"  - {fail['url']} : {fail['reason']}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python ingest_wikisource.py wikisource_urls.txt")
        sys.exit(1)

    urls_file = sys.argv[1]
    urls = read_urls(urls_file)
    logger.info(f"{len(urls)} URL(s) trouvée(s) dans {urls_file}")

    ingest_wikisource_urls(urls)