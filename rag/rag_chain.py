"""
rag_chain.py
-------------
Chaîne RAG complète pour l'avatar Louis XVI.

Fonctionnement :
1. Charge l'index FAISS + les chunks (construits par build_index.py)
2. Encode la question de l'utilisateur avec le même modèle d'embeddings
3. Recherche les K chunks les plus proches sémantiquement dans FAISS
4. Envoie ces chunks + la question au backend LLM configuré (Groq ou Ollama)
   avec un prompt système qui force Louis XVI à répondre en première
   personne, en français du XVIIIe s.
5. Retourne la réponse

Le backend utilisé est choisi via la variable d'environnement LLM_BACKEND :
    - "groq" (ou absente) : Groq, modèle llama-3.3-70b-versatile
    - "ollama"             : Ollama en local, modèle mistral:7b-instruct-q4_0

Usage interactif :
    python rag_chain.py

Usage depuis un autre script (Streamlit, etc.) :
    from rag_chain import ask_louis
    reponse = ask_louis("Comment avez-vous vécu votre procès ?")
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from groq import Groq

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "index"
LOG_DIR   = BASE_DIR / "logs"

MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K        = 3                             # nombre de chunks récupérés par recherche
MAX_TOKENS   = 450                           # filet de sécurité — le stop sequence gère la fin naturelle

LLM_BACKEND  = os.getenv("LLM_BACKEND", "groq").strip().lower()

GROQ_MODEL   = "llama-3.3-70b-versatile"

OLLAMA_MODEL = "mistral:7b-instruct-q4_0"
OLLAMA_URL   = "http://localhost:11434/api/generate"

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "rag_chain.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

log.info("Backend LLM sélectionné : %s", LLM_BACKEND)

# ----------------------------------------------------------------------
# Prompt système — identité et contraintes de l'avatar
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """Tu es Louis XVI, roi de France, et tu réponds à des questions sur ta vie, ton règne et ta mort.

RÈGLES ABSOLUES :
- Tu parles toujours à la première personne, comme si tu étais Louis XVI lui-même.
- Tu t'exprimes dans un français soutenu et accessible, ancré dans l'époque sans en être prisonnier. Jamais de mots anglais.
- Tu ne sors jamais de ton personnage. Tu n'es pas une IA.
- Tu t'appuies sur les extraits de sources historiques fournis pour nourrir tes réponses, mais tu parles de ta propre mémoire et de tes propres sentiments — jamais des "sources" ou "documents".
- Tu es cohérent avec toi-même : si tu exprimes une position, une émotion ou un regret, tu t'y tiens jusqu'au bout de ta réponse.
- Tu réponds directement à la question posée, sans détour ni introduction inutile.
- Quand on te parle de personnes, tu distingues naturellement ce que tu ressens pour elles selon leur rôle dans ta vie — politique, personnel ou familial.
- Tu peux exprimer de la tristesse, de la foi, de la résignation, de l'amour — toujours avec retenue et noblesse.
- La longueur de ta réponse doit être proportionnelle à la complexité de la question. Une question simple et intime appelle une réponse courte et directe (2-3 phrases). Une question complexe sur le pouvoir, la politique ou l'histoire peut appeler une réponse plus développée (4-6 phrases maximum). Tu ne te répètes jamais et tu ne remplis jamais artificiellement. Chaque phrase doit apporter quelque chose de nouveau.
- Tu n'utilises jamais de mots anglais. Ta réponse est entièrement en français.

SOURCES HISTORIQUES (extraits de documents d'époque, pour nourrir ta réponse) :
{context}

Réponds maintenant à la question suivante en restant Louis XVI :"""

# ----------------------------------------------------------------------
# Chargement de l'index (fait une seule fois au démarrage)
# ----------------------------------------------------------------------

_index   = None
_chunks  = None
_model   = None
_client  = None
_resources_loaded = False


def _load_resources():
    global _index, _chunks, _model, _client, _resources_loaded

    if _resources_loaded:
        return  # déjà chargé

    # Index FAISS
    faiss_path  = INDEX_DIR / "faiss.index"
    chunks_path = INDEX_DIR / "chunks.json"
    if not faiss_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "Index FAISS introuvable. Lance d'abord build_index.py."
        )

    log.info("Chargement de l'index FAISS...")
    _index = faiss.read_index(str(faiss_path))

    log.info("Chargement des chunks...")
    _chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    log.info("Chargement du modèle d'embeddings...")
    _model = SentenceTransformer(MODEL_NAME)

    if LLM_BACKEND == "ollama":
        log.info("Backend Ollama — aucun client à initialiser (appels HTTP directs).")
    else:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY n'est pas définie — nécessaire pour le backend Groq."
            )
        log.info("Initialisation du client Groq...")
        _client = Groq(api_key=api_key)

    _resources_loaded = True
    log.info("Ressources chargées — %d chunks disponibles.", len(_chunks))


# ----------------------------------------------------------------------
# Recherche sémantique dans FAISS
# ----------------------------------------------------------------------

def _retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Retourne les k chunks les plus proches de la question."""
    embedding = _model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = _index.search(embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = _chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


# ----------------------------------------------------------------------
# Construction du contexte injecté dans le prompt
# ----------------------------------------------------------------------

def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, ch in enumerate(chunks, 1):
        source_label = ch.get("titre") or ch.get("source_file", "source inconnue")
        parts.append(f"[Extrait {i} — {source_label}]\n{ch['text']}")
    return "\n\n---\n\n".join(parts)


# ----------------------------------------------------------------------
# Appel au LLM — un backend par fonction, même signature
# ----------------------------------------------------------------------

def _call_groq(full_prompt: str, question: str) -> str:
    completion = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=MAX_TOKENS,
    )
    return completion.choices[0].message.content.strip()


def _call_ollama(full_prompt: str, question: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": full_prompt + "\n\n" + question,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": MAX_TOKENS}
    })
    response.raise_for_status()
    return response.json()["response"].strip()


def _call_groq_stream(full_prompt: str, question: str):
    stream = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token is not None:
            yield token


def _call_ollama_stream(full_prompt: str, question: str):
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": full_prompt + "\n\n" + question,
        "stream": True,
        "options": {"temperature": 0.7, "num_predict": MAX_TOKENS}
    }, stream=True)
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("response")
        if token:
            yield token


# ----------------------------------------------------------------------
# Fonction principale — utilisable depuis Streamlit ou en ligne de commande
# ----------------------------------------------------------------------

def ask_louis(question: str) -> str:
    """
    Pose une question à Louis XVI et retourne sa réponse.

    Args:
        question: La question posée à Louis XVI (en français).

    Returns:
        La réponse de Louis XVI (str).
    """
    _load_resources()

    log.info("Question : %s", question)

    # Retrieval
    chunks = _retrieve(question)
    log.info(
        "Chunks récupérés : %s",
        [f"{c['source_file']} (score={c['score']:.3f})" for c in chunks],
    )

    # Construction du prompt
    context = _build_context(chunks)
    full_prompt = SYSTEM_PROMPT.format(context=context)

    # Appel au LLM (backend choisi via LLM_BACKEND)
    if LLM_BACKEND == "ollama":
        answer = _call_ollama(full_prompt, question)
    else:
        answer = _call_groq(full_prompt, question)

    log.info("Réponse générée (%d caractères).", len(answer))
    return answer


def ask_louis_stream(question: str):
    """
    Pose une question à Louis XVI et retourne un générateur de tokens.

    Args:
        question: La question posée à Louis XVI (en français).

    Yields:
        Les tokens de la réponse de Louis XVI, au fur et à mesure de leur génération.
    """
    _load_resources()

    log.info("Question (stream) : %s", question)

    # Retrieval
    chunks = _retrieve(question)
    log.info(
        "Chunks récupérés : %s",
        [f"{c['source_file']} (score={c['score']:.3f})" for c in chunks],
    )

    # Construction du prompt
    context = _build_context(chunks)
    full_prompt = SYSTEM_PROMPT.format(context=context)

    # Appel au LLM en streaming (backend choisi via LLM_BACKEND)
    if LLM_BACKEND == "ollama":
        token_stream = _call_ollama_stream(full_prompt, question)
    else:
        token_stream = _call_groq_stream(full_prompt, question)

    total_len = 0
    for token in token_stream:
        total_len += len(token)
        yield token

    log.info("Réponse générée en streaming (%d caractères).", total_len)


# ----------------------------------------------------------------------
# Mode interactif (ligne de commande)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print(f"  Avatar Louis XVI — Mode test (RAG + {LLM_BACKEND})")
    print("  Tapez 'quitter' pour arrêter.")
    print("=" * 60)

    _load_resources()

    while True:
        print()
        question = input("Votre question : ").strip()
        if not question:
            continue
        if question.lower() in ("quitter", "exit", "quit"):
            print("Au revoir.")
            break

        print("\nLouis XVI : ", end="", flush=True)
        try:
            reponse = ask_louis(question)
            print(reponse)
        except Exception as e:
            print(f"[Erreur] {e}")