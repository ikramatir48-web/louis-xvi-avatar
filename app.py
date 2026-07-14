"""
app.py
------
Interface Streamlit pour l'avatar Louis XVI.

Lancement (depuis la racine du projet) :
    streamlit run app.py
"""

import logging
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "rag"))

# ----------------------------------------------------------------------
# Logging — console + expander Streamlit pour déboguer sur le cloud
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

if "log_lines" not in st.session_state:
    st.session_state.log_lines = []


class _SessionLogHandler(logging.Handler):
    def emit(self, record):
        st.session_state.log_lines.append(self.format(record))


if not any(isinstance(h, _SessionLogHandler) for h in logger.handlers):
    _session_handler = _SessionLogHandler()
    _session_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_session_handler)

# ----------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Interview de Louis XVI",
    page_icon="👑",
    layout="centered",
)

# ----------------------------------------------------------------------
# Style — thème sombre, dorures et bordeaux
# ----------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1a1210;
        color: #f0e6d2;
    }
    h1, h2, h3 {
        color: #d4af37 !important;
        font-family: Georgia, 'Times New Roman', serif;
    }
    .stTextInput input {
        background-color: #2a1d1a;
        color: #f0e6d2;
        border: 1px solid #d4af37;
    }
    .stButton button {
        background-color: #722f37;
        color: #f0e6d2;
        border: 1px solid #d4af37;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #8a3a44;
        border: 1px solid #f0e6d2;
        color: #f0e6d2;
    }
    .question-box {
        background-color: #2a1d1a;
        border-left: 4px solid #d4af37;
        padding: 0.8em 1em;
        margin: 1em 0 0.4em 0;
        border-radius: 4px;
        color: #f0e6d2;
    }
    .louis-box {
        background-color: #3a1f24;
        border: 1px solid #d4af37;
        padding: 1em 1.2em;
        margin-bottom: 1.2em;
        border-radius: 6px;
        color: #f0e6d2;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# En-tête
# ----------------------------------------------------------------------

st.title("👑 Interview de Louis XVI")
st.markdown(
    "Posez vos questions à Louis XVI et recevez ses réponses, "
    "puisées dans les archives et documents de son époque."
)

# ----------------------------------------------------------------------
# Configuration du backend LLM (lue depuis l'environnement)
# ----------------------------------------------------------------------

LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq").strip().lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if LLM_BACKEND == "groq" and not GROQ_API_KEY:
    st.warning(
        "Backend Groq sélectionné mais la variable d'environnement "
        "GROQ_API_KEY n'est pas définie. L'appel à Louis XVI échouera."
    )

logger.info(
    "App démarrée, LLM_BACKEND=%s, GROQ_API_KEY=%s",
    LLM_BACKEND,
    "définie" if GROQ_API_KEY else "manquante",
)

# ----------------------------------------------------------------------
# Import de la chaîne RAG — ne doit jamais faire planter l'app
# ----------------------------------------------------------------------

try:
    from rag_chain import ask_louis_stream
    import_error = None
except Exception as e:
    ask_louis_stream = None
    import_error = str(e)
    logger.exception("Échec de l'import de rag_chain")

if import_error:
    st.error(f"Impossible de charger l'avatar Louis XVI : {import_error}")

# ----------------------------------------------------------------------
# Historique de la conversation (persiste pendant la session)
# ----------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # liste de (question, reponse, erreur)

# ----------------------------------------------------------------------
# Formulaire de question
# ----------------------------------------------------------------------

with st.form("question_form", clear_on_submit=True):
    question = st.text_input("Votre question à Louis XVI :")
    submitted = st.form_submit_button("Poser la question")


def _count_tokens(token_stream):
    count = 0
    for token in token_stream:
        count += 1
        yield token
    logger.info("Stream terminé, %d tokens reçus", count)


if submitted:
    question = question.strip()
    if not question:
        st.warning("Veuillez saisir une question.")
    elif ask_louis_stream is None:
        st.error("L'avatar Louis XVI n'est pas disponible (voir l'erreur ci-dessus).")
    else:
        try:
            logger.info("Appel ask_louis_stream, question=%s", question)
            reponse = st.write_stream(_count_tokens(ask_louis_stream(question)))
            st.session_state.history.append((question, reponse, None))
        except Exception as e:
            logger.exception("Erreur lors de la génération de la réponse")
            st.session_state.history.append((question, None, str(e)))

# ----------------------------------------------------------------------
# Affichage de la conversation (la plus récente en premier)
# ----------------------------------------------------------------------

if st.session_state.history:
    st.markdown("### Conversation")
    for q, reponse, erreur in reversed(st.session_state.history):
        st.markdown(f'<div class="question-box">❓ {q}</div>', unsafe_allow_html=True)
        if erreur:
            st.error(f"Une erreur est survenue lors de la génération de la réponse : {erreur}")
        else:
            st.markdown(f'<div class="louis-box">👑 {reponse}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Logs techniques (pour déboguer sans regarder la console)
# ----------------------------------------------------------------------

with st.expander("Logs techniques"):
    st.code("\n".join(st.session_state.log_lines) or "Aucun log pour le moment.")
