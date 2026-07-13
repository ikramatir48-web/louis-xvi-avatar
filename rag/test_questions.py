"""
test_questions.py
------------------
Pose automatiquement toutes les questions de questions.txt à l'avatar
Louis XVI (via ask_louis) et enregistre les réponses dans
logs/resultats_questions.txt.

Usage (depuis le dossier rag/) :
    python test_questions.py
"""

import time
from pathlib import Path

from rag_chain import ask_louis

BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = BASE_DIR / "questions.txt"
OUTPUT_FILE = BASE_DIR / "logs" / "resultats_questions.txt"

DELAI_ENTRE_QUESTIONS = 3  # secondes, pour ne pas surcharger l'API Groq

SEPARATEUR = "═" * 62


def charger_questions(path: Path) -> list[tuple[str, str]]:
    """Retourne une liste de (section, question), en ignorant les lignes vides et les titres."""
    questions = []
    section_courante = ""
    for ligne in path.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        if ligne.startswith("#"):
            section_courante = ligne.lstrip("#").strip()
            continue
        questions.append((section_courante, ligne))
    return questions


def main():
    questions = charger_questions(QUESTIONS_FILE)
    total = len(questions)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for i, (section, question) in enumerate(questions, 1):
            print(f"⏳ Question {i}/{total} en cours...")

            f.write(f"{SEPARATEUR}\n")
            f.write(f"Question {i}/{total} — {section}\n")
            f.write(f"{SEPARATEUR}\n")
            f.write(f"❓ {question}\n\n")
            f.write("👑 Louis XVI :\n")

            try:
                reponse = ask_louis(question)
                f.write(f"{reponse}\n\n")
            except Exception as e:
                f.write(f"[ERREUR] {e}\n\n")

            f.flush()

            if i < total:
                time.sleep(DELAI_ENTRE_QUESTIONS)

    print(f"\n✅ Terminé. Résultats écrits dans {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
