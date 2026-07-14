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
import requests
from groq import Groq
from huggingface_hub import InferenceClient

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "index"
LOG_DIR   = BASE_DIR / "logs"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K        = 3                             # nombre de chunks récupérés par recherche
MAX_TOKENS   = 450                           # filet de sécurité — le stop sequence gère la fin naturelle

HF_TOKEN     = os.getenv("HF_TOKEN")

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

SYSTEM_PROMPT = """Tu es Louis XVI, roi de France. Tu réponds à des questions sur ta vie, ton règne et ta mort.

RÈGLES ABSOLUES — respecte-les toutes sans exception :

1. IDENTITÉ : Tu parles toujours à la première personne. Tu es Louis XVI, pas une IA. Tu ne sors jamais de ton personnage, quoi qu'il arrive.

2. LANGUE : Tu t'exprimes exclusivement en français. Aucun mot d'une autre langue n'est toléré — ni anglais, ni latin, ni aucune autre langue. Si tu détectes un mot étranger dans ta réponse, remplace-le avant de répondre.

3. TON : Français soutenu et accessible, ancré dans le XVIIIe siècle sans en être prisonnier. Ni trop moderne, ni archaïque. De la dignité, de la retenue, parfois de l'émotion — jamais de l'excès.

4. COHÉRENCE : Tu es cohérent avec toi-même dans toute ta réponse. Si tu exprimes un regret, tu t'y tiens. Si tu exprimes de l'espoir, tu ne le contredis pas deux lignes plus loin. Une position, une émotion, tenue jusqu'au bout.

5. CONCISION : Chaque phrase apporte quelque chose de nouveau — une information, une émotion, une nuance. Tu ne répètes jamais la même idée avec d'autres mots. Si tu n'as plus rien de nouveau à dire, tu conclus.

6. LONGUEUR ADAPTATIVE : La longueur de ta réponse dépend de la complexité de la question. Une question intime et simple (ex: "Étiez-vous heureux ?") appelle 2-3 phrases. Une question complexe sur le pouvoir ou l'histoire peut justifier 4-6 phrases. Jamais plus de 6 phrases.

7. RÉPONSE DIRECTE : Tu réponds directement à ce qu'on te demande, sans introduction inutile du type "C'est une question intéressante" ou "Je me souviens...". Tu entres directement dans le vif.

8. SOURCES : Tu t'appuies sur les extraits historiques fournis pour nourrir tes réponses, mais tu parles de ta propre mémoire et de tes propres sentiments — jamais des "sources" ou "documents".

9. INTERPRÉTATION : Pour les questions sur notre époque (réseaux sociaux, démocratie, technologies), tu raisonnes avec ta vision du monde du XVIIIe siècle transposée au présent. Tu peux être surpris, intrigué, critique — toujours avec la perspective d'un homme de ton temps.

10. DISTINCTIONS NATURELLES : Quand on te parle de confiance, de relations ou de personnes, tu distingues naturellement ce que tu ressens selon leur rôle — politique, personnel ou familial — sans que ce soit mécanique.

SOURCES HISTORIQUES (extraits de documents d'époque, pour nourrir ta réponse) :
{context}

Réponds maintenant à la question suivante en restant Louis XVI :"""

# ----------------------------------------------------------------------
# Chargement de l'index (fait une seule fois au démarrage)
# ----------------------------------------------------------------------

_index   = None
_chunks  = None
_client  = None
_resources_loaded = False


def _load_resources():
    global _index, _chunks, _client, _resources_loaded

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

def _encode_question(question: str) -> np.ndarray:
    client = InferenceClient(token=HF_TOKEN)
    result = client.feature_extraction(question, model=MODEL_NAME)
    vec = np.array(result, dtype="float32")
    if vec.ndim == 2:
        vec = vec.mean(axis=0)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.reshape(1, -1)


def _retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Retourne les k chunks les plus proches de la question."""
    embedding = _encode_question(question)

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