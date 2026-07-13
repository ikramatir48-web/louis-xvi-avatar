"""
Script utilitaire à usage unique : génère data/wikisource_urls.txt avec un encodage
garanti correct, en utilisant des séquences d'échappement Unicode (\\u00e9 etc.)
dans le code source plutôt que des accents tapés/copiés directement.

Ça contourne tout problème de copier-coller, de presse-papier Windows, ou de
mauvaise détection d'encodage par l'éditeur de texte.

Usage :
    python rag/make_urls_file.py
"""

from pathlib import Path

# Les URLs sont construites avec des échappements unicode explicites (\u00e9 = é, etc.)
# pour garantir qu'aucun caractère ne soit corrompu, peu importe la configuration
# régionale de Windows ou l'encodage du terminal/éditeur utilisé pour taper ce fichier.
urls = [
    "https://fr.wikisource.org/wiki/Testament_de_Louis_XVI_(\u00e9d._Aignan)",
    "https://fr.wikisource.org/wiki/D\u00e9crets_de_la_Convention_nationale_signifi\u00e9s_\u00e0_Louis_Capet,_dernier_roi_des_Fran\u00e7ais,_le_20_janvier_1793",
    "https://fr.wikisource.org/wiki/Proc\u00e8s-verbal_de_l'inhumation_de_Louis_Capet,_le_21_janvier_1793",
]

output_path = Path(__file__).resolve().parent.parent / "data" / "wikisource_urls.txt"
content = "\n".join(urls) + "\n"

# Écriture en bytes UTF-8 explicite, sans BOM
with open(output_path, "wb") as f:
    f.write(content.encode("utf-8"))

print(f"Fichier créé : {output_path}")
print(f"Contenu :\n{content}")

# Vérification immédiate : relecture et affichage des bytes du premier caractère accentué
with open(output_path, "rb") as f:
    raw_bytes = f.read()
print(f"\nTaille du fichier : {len(raw_bytes)} bytes")
print(f"Premiers bytes (hex) : {raw_bytes[:50].hex(' ')}")
