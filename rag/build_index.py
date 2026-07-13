"""
build_index.py
----------------
Construit l'index de recherche sémantique (FAISS) à partir des fichiers
nettoyés dans data/cleaned/.

Étapes :
1. Lecture de tous les .txt dans data/cleaned/
2. Découpage en chunks (par paragraphes, ~300-500 mots, avec léger chevauchement)
3. Génération des embeddings (sentence-transformers, modèle multilingue)
4. Construction et sauvegarde de l'index FAISS + métadonnées associées

Sorties :
- data/index/faiss.index       (index vectoriel binaire)
- data/index/chunks.json       (texte + métadonnées de chaque chunk, même ordre que l'index)

Usage :
    python build_index.py
"""

import os
import re
import json
import logging
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
INDEX_DIR = BASE_DIR / "data" / "index"
LOG_DIR = BASE_DIR / "logs"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

TARGET_WORDS = 400       # taille cible d'un chunk (en mots)
MAX_WORDS = 550          # taille max avant découpe forcée
MIN_WORDS = 60           # en dessous, on fusionne avec le paragraphe suivant
OVERLAP_WORDS = 50       # chevauchement entre deux chunks consécutifs d'un même document

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "build_index.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Extraction du header [SOURCE: ...] etc. en haut de chaque fichier
# ----------------------------------------------------------------------

HEADER_LINE_RE = re.compile(r"^\[([A-ZÀ-Ü _]+):\s*(.*)\]\s*$")


def extract_header_and_body(text: str) -> tuple[dict, str]:
    """
    Sépare les lignes d'en-tête de type [SOURCE: ...] / [TITRE: ...] etc.
    du corps du texte. Retourne (métadonnées, corps_du_texte).
    """
    lines = text.split("\n")
    meta: dict[str, str] = {}
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "---":
            body_start = i + 1
            break
        m = HEADER_LINE_RE.match(stripped)
        if m:
            key, value = m.group(1).strip(), m.group(2).strip()
            meta[key] = value
            body_start = i + 1
        elif stripped == "" and meta:
            # ligne vide après des headers déjà trouvés : on continue
            body_start = i + 1
            continue
        elif stripped == "":
            continue
        else:
            # première ligne de contenu réel
            body_start = i
            break

    body = "\n".join(lines[body_start:]).strip()
    return meta, body


# ----------------------------------------------------------------------
# Découpage en chunks
# ----------------------------------------------------------------------

def split_into_paragraphs(text: str) -> list[str]:
    """Découpe le texte en paragraphes sur les doubles sauts de ligne."""
    raw_paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw_paragraphs if p.strip()]


def word_count(s: str) -> int:
    return len(s.split())


def chunk_document(body: str) -> list[str]:
    """
    Regroupe les paragraphes en chunks autour de TARGET_WORDS mots,
    avec un chevauchement léger entre chunks consécutifs pour préserver
    le contexte aux frontières.
    """
    paragraphs = split_into_paragraphs(body)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = word_count(para)

        # Paragraphe à lui seul plus gros que MAX_WORDS : on le découpe en phrases
        if para_words > MAX_WORDS:
            if current:
                chunks.append("\n\n".join(current))
                current, current_words = [], 0
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub_current: list[str] = []
            sub_words = 0
            for sent in sentences:
                sw = word_count(sent)
                if sub_words + sw > TARGET_WORDS and sub_current:
                    chunks.append(" ".join(sub_current))
                    sub_current, sub_words = [], 0
                sub_current.append(sent)
                sub_words += sw
            if sub_current:
                chunks.append(" ".join(sub_current))
            continue

        # Ajouter le paragraphe au chunk courant tant qu'on ne dépasse pas la cible
        if current_words + para_words <= TARGET_WORDS or current_words < MIN_WORDS:
            current.append(para)
            current_words += para_words
        else:
            chunks.append("\n\n".join(current))
            current = [para]
            current_words = para_words

    if current:
        # Fusionner un dernier chunk trop petit avec le précédent si possible
        if current_words < MIN_WORDS and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(current)
        else:
            chunks.append("\n\n".join(current))

    # Ajout d'un léger chevauchement : on préfixe chaque chunk (sauf le premier)
    # avec la fin du chunk précédent, pour ne pas perdre le contexte aux jointures.
    overlapped: list[str] = []
    for i, ch in enumerate(chunks):
        if i == 0:
            overlapped.append(ch)
            continue
        prev_words = chunks[i - 1].split()
        tail = " ".join(prev_words[-OVERLAP_WORDS:]) if len(prev_words) > OVERLAP_WORDS else chunks[i - 1]
        overlapped.append(tail + "\n\n" + ch)

    return overlapped


# ----------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------

def build():
    if not CLEANED_DIR.exists():
        log.error("Le dossier %s n'existe pas. Lancez d'abord clean_text.py.", CLEANED_DIR)
        return

    txt_files = sorted(CLEANED_DIR.glob("*.txt"))
    if not txt_files:
        log.error("Aucun fichier .txt trouvé dans %s", CLEANED_DIR)
        return

    log.info("%d fichier(s) source(s) trouvé(s) dans data/cleaned/", len(txt_files))

    all_chunks: list[dict] = []

    for path in txt_files:
        raw = path.read_text(encoding="utf-8")
        meta, body = extract_header_and_body(raw)
        doc_chunks = chunk_document(body)

        if not doc_chunks:
            log.warning("Aucun chunk produit pour %s (fichier vide après extraction du header ?)", path.name)
            continue

        for idx, chunk_text in enumerate(doc_chunks):
            all_chunks.append({
                "id": f"{path.stem}__chunk{idx:03d}",
                "source_file": path.name,
                "source": meta.get("SOURCE", ""),
                "titre": meta.get("TITRE", ""),
                "auteur": meta.get("AUTEUR", ""),
                "origine": meta.get("ORIGINE", ""),
                "type": meta.get("TYPE", ""),
                "chunk_index": idx,
                "n_chunks_in_doc": len(doc_chunks),
                "text": chunk_text,
                "word_count": word_count(chunk_text),
            })

        log.info("  %-55s -> %2d chunk(s)", path.name, len(doc_chunks))

    if not all_chunks:
        log.error("Aucun chunk produit sur l'ensemble du corpus. Abandon.")
        return

    log.info("Total : %d chunks sur %d documents", len(all_chunks), len(txt_files))

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    log.info("Chargement du modèle d'embeddings : %s (premier lancement = téléchargement, ~470 Mo)", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    log.info("Calcul des embeddings pour %d chunks...", len(texts))
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # nécessaire pour utiliser un index à similarité cosinus
    )
    embeddings = embeddings.astype("float32")

    # ------------------------------------------------------------------
    # Index FAISS (produit scalaire = cosinus, car embeddings normalisés)
    # ------------------------------------------------------------------
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("Index FAISS construit : %d vecteurs, dimension %d", index.ntotal, dim)

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss_path = INDEX_DIR / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    log.info("Index FAISS sauvegardé : %s", faiss_path)

    chunks_path = INDEX_DIR / "chunks.json"
    with open(chunks_path, "wb") as f:
        f.write(json.dumps(all_chunks, ensure_ascii=False, indent=2).encode("utf-8"))
    log.info("Métadonnées des chunks sauvegardées : %s", chunks_path)

    meta_summary = {
        "model_name": MODEL_NAME,
        "embedding_dim": dim,
        "n_chunks": len(all_chunks),
        "n_documents": len(txt_files),
        "target_words_per_chunk": TARGET_WORDS,
        "overlap_words": OVERLAP_WORDS,
    }
    summary_path = INDEX_DIR / "index_info.json"
    with open(summary_path, "wb") as f:
        f.write(json.dumps(meta_summary, ensure_ascii=False, indent=2).encode("utf-8"))
    log.info("Résumé de l'index sauvegardé : %s", summary_path)

    log.info("-" * 60)
    log.info("Construction de l'index terminée avec succès.")
    log.info("%d chunks indexés depuis %d documents.", len(all_chunks), len(txt_files))


if __name__ == "__main__":
    build()
