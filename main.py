import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1"}
)
MODEL = "gemini-1.5-flash"

app = FastAPI(title="AutoInspect IA - API Gratuite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INSPECTION_PROMPT = """
Tu es un expert en inspection automobile avec 20 ans d'experience en carrosserie et mecanique.
Analyse cette photo de vehicule avec une precision maximale et detecte TOUS les defauts visibles.

Defauts a rechercher :
- Bosses, pocs, renfoncements
- Rayures, griffures, eraflures
- Coulures de peinture, peeling, differences de teinte
- Jantes abimees, voilees, rayees
- Ecarts de fitment (portes, capot, coffre mal alignes)
- Oxydation, rouille
- Fissures sur pare-chocs
- Impacts sur vitres
- Traces de reparations anterieures

Reponds UNIQUEMENT en JSON valide sans markdown :
{
  "defauts": [
    {
      "type": "string",
      "localisation": "string",
      "severite": "mineur|modere|grave",
      "description": "string",
      "cout_reparation_min": 0,
      "cout_reparation_max": 0
    }
  ],
  "score_global": 100,
  "resume": "string",
  "recommandations": ["string"]
}
"""

VEHICLE_PROMPT = """
Identifie ce vehicule. Reponds UNIQUEMENT en JSON :
{
  "marque": "string",
  "modele": "string",
  "couleur": "string",
  "type": "string"
}
"""


def parse_json(text: str) -> dict:
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
    return {"status": "ok", "model": MODEL, "tier": "gratuit"}


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
                errors.append(f"{file.filename}: trop lourd")
                continue

            image_part = types.Part.from_bytes(
                data=content,
                mime_type=file.content_type
            )

            # Identification vehicule sur 1ere photo
            if i == 0:
                try:
                    info_resp = client.models.generate_content(
                        model=MODEL,
                        contents=[VEHICLE_PROMPT, image_part]
                    )
                    vehicle_info = parse_json(info_resp.text)
                except Exception:
                    vehicle_info = {"marque": "Inconnu", "modele": "Inconnu", "couleur": "Inconnu", "type": "Inconnu"}

            # Analyse des defauts
            resp = client.models.generate_content(
                model=MODEL,
                contents=[INSPECTION_PROMPT, image_part]
            )
            result = parse_json(resp.text)

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
        raise HTTPException(status_code=500, detail=f"Aucune image analysee. Erreurs: {errors}")

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
