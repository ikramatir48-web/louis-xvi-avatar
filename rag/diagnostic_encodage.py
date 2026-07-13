"""
Script de diagnostic d'encodage — à lancer une seule fois pour identifier
précisément pourquoi les accents sont corrompus (mojibake) sur cette machine.

Usage :
    python diagnostic_encodage.py

Affiche des informations système ET fait un vrai test contre l'API Wikisource,
pour localiser exactement à quelle étape le texte se corrompt.
"""

import sys
import locale
import json
import requests

print("=" * 60)
print("DIAGNOSTIC D'ENCODAGE")
print("=" * 60)

print(f"\n[1] Encodage par défaut du système (locale) : {locale.getpreferredencoding()}")
print(f"[2] Encodage de sys.stdout : {sys.stdout.encoding}")
print(f"[3] Version Python : {sys.version}")

print("\n[4] Test d'écriture/lecture d'un fichier avec accents...")
texte_test = "l'autorité, Décrets, procès, été"
with open("test_encodage.txt", "wb") as f:
    f.write(texte_test.encode("utf-8"))
with open("test_encodage.txt", "rb") as f:
    relu = f.read().decode("utf-8")
print(f"    Texte original : {texte_test}")
print(f"    Texte relu     : {relu}")
print(f"    Identiques : {texte_test == relu}")

print("\n[5] Test réel contre l'API Wikisource...")
try:
    api_url = "https://fr.wikisource.org/w/api.php"
    params = {
        "action": "parse",
        "page": "Testament_de_Louis_XVI_(éd._Aignan)",
        "prop": "text",
        "format": "json",
        "formatversion": "2",
        "redirects": True
    }
    headers = {"User-Agent": "DiagnosticBot/1.0"}
    response = requests.get(api_url, params=params, headers=headers, timeout=15)

    print(f"    Content-Type de la réponse : {response.headers.get('Content-Type')}")
    print(f"    Encodage deviné par requests : {response.encoding}")
    print(f"    Encodage apparent (chardet) : {response.apparent_encoding}")

    # Méthode A : response.json() standard
    data_a = response.json()
    html_a = data_a.get("parse", {}).get("text", "")
    extrait_a = html_a[:200] if html_a else "VIDE"

    # Méthode B : décodage manuel des bytes bruts
    data_b = json.loads(response.content.decode("utf-8"))
    html_b = data_b.get("parse", {}).get("text", "")
    extrait_b = html_b[:200] if html_b else "VIDE"

    print(f"\n    [Méthode A - response.json()] Extrait :")
    print(f"    {extrait_a}")
    print(f"\n    [Méthode B - decode utf-8 manuel] Extrait :")
    print(f"    {extrait_b}")
    print(f"\n    Les deux méthodes donnent le même résultat : {extrait_a == extrait_b}")

except Exception as e:
    print(f"    Erreur lors du test réseau : {e}")

print("\n" + "=" * 60)
print("Copie-colle TOUT ce résultat dans le chat pour qu'on identifie le problème.")
print("=" * 60)