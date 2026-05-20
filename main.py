import os
import base64
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image
import io

# Configuration Gemini (GRATUIT - 1500 requêtes/jour)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI(title="AutoInspect IA - API Gratuite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INSPECTION_PROMPT = """
Tu es un expert en inspection automobile avec 20 ans d'expérience en carrosserie et mécanique.
Analyse cette photo de véhicule avec une précision maximale et détecte TOUS les défauts visibles.

Défauts à rechercher impérativement :
- Bosses, pocs, renfoncements (même minimes)
- Rayures, griffures, éraflures sur la carrosserie
- Coulures de peinture, peeling, différences de teinte
- Jantes abîmées, voilées, rayées, ébréchées
- Écarts de fitment (portes, capot, coffre mal alignés)
- Oxydation, rouille, bulles sous la peinture
- Fissures, craquelures (pare-chocs, plastiques)
- Impacts sur vitres, pare-brise
- Dommages sur rétroviseurs, poignées, joints
- Pneus usés, déformés, sous-gonflés
- Traces de réparations antérieures (masticage, repeinture)

Réponds UNIQUEMENT en JSON valide sans aucun texte autour, sans markdown, avec exactement ce format :
{
  "defauts": [
    {
      "type": "string (ex: Bosse, Rayure profonde, Coulure de peinture...)",
      "localisation": "string (ex: Aile avant gauche, Porte arrière droite...)",
      "severite": "mineur|modere|grave",
      "description": "string (description précise)",
      "cout_reparation_min": 0,
      "cout_reparation_max": 0
    }
  ],
  "score_global": 100,
  "resume": "string",
  "recommandations": ["string"]
}

Si aucun défaut n'est visible, retourne un tableau vide et un score de 95-100.
"""

VEHICLE_PROMPT = """
Identifie ce véhicule et réponds UNIQUEMENT en JSON sans texte autour :
{
  "marque": "string ou Inconnu",
  "modele": "string ou Inconnu",
  "couleur": "string",
  "type": "string (berline, SUV, break, coupe, etc.)"
}
"""


def load_image(content: bytes) -> Image.Image:
    return Image.open(io.BytesIO(content))


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    return json.loads(text)


@app.get("/health")
def health():
    return {"status": "ok", "model": "gemini-1.5-flash", "tier": "gratuit"}


@app.post("/inspect")
async def inspect_vehicle(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 photos")

    all_defauts = []
    all_recommandations = set()
    scores = []
    vehicle_info = None
    errors = []

    for i, file in enumerate(files):
        if not file.content_type.startswith("image/"):
            errors.append(f"{file.filename}: pas une image")
            continue

        try:
            content = await file.read()
            if len(content) > 20 * 1024 * 1024:
                errors.append(f"{file.filename}: trop lourd (max 20MB)")
                continue

            img = load_image(content)

            if i == 0:
                try:
                    info_resp = model.generate_content([VEHICLE_PROMPT, img])
                    vehicle_info = parse_json_response(info_resp.text)
                except Exception:
                    vehicle_info = {"marque": "Inconnu", "modele": "Inconnu", "couleur": "Inconnu", "type": "Inconnu"}

            resp = model.generate_content([INSPECTION_PROMPT, img])
            result = parse_json_response(resp.text)

            for defaut in result.get("defauts", []):
                defaut["photo_source"] = file.filename
                all_defauts.append(defaut)

            for rec in result.get("recommandations", []):
                all_recommandations.add(rec)

            scores.append(result.get("score_global", 100))

        except json.JSONDecodeError as e:
            errors.append(f"{file.filename}: erreur JSON - {str(e)}")
        except Exception as e:
            errors.append(f"{file.filename}: erreur - {str(e)}")

    if not scores:
        raise HTTPException(status_code=500, detail=f"Aucune image analysée. Erreurs: {errors}")

    score_final = round(sum(scores) / len(scores))
    cout_min = sum(d.get("cout_reparation_min", 0) for d in all_defauts)
    cout_max = sum(d.get("cout_reparation_max", 0) for d in all_defauts)
    severite_order = {"grave": 0, "modere": 1, "mineur": 2}
    all_defauts.sort(key=lambda x: severite_order.get(x.get("severite", "mineur"), 3))

    return {
        "score_global": score_final,
        "vehicule": vehicle_info,
        "defauts": all_defauts,
        "nb_defauts": len(all_defauts),
        "nb_graves": len([d for d in all_defauts if d.get("severite") == "grave"]),
        "nb_moderes": len([d for d in all_defauts if d.get("severite") == "modere"]),
        "nb_mineurs": len([d for d in all_defauts if d.get("severite") == "mineur"]),
        "cout_reparation_min": cout_min,
        "cout_reparation_max": cout_max,
        "recommandations": list(all_recommandations),
        "photos_analysees": len(scores),
        "erreurs": errors if errors else None,
    }
