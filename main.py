import os
import base64
import json
import time
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

app = FastAPI(title="AutoInspect IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INSPECTION_PROMPT = """Tu es un expert en inspection automobile. Analyse cette photo et detecte TOUS les defauts visibles : bosses, rayures, coulures de peinture, jantes abimees, ecarts de fitment, oxydation, fissures, traces de reparations.

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
}"""

VEHICLE_PROMPT = """Identifie ce vehicule. Reponds UNIQUEMENT en JSON :
{"marque": "string", "modele": "string", "couleur": "string", "type": "string"}"""


def call_gemini(prompt, image_bytes, mime_type):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": image_b64}}
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048}
    }
    resp = requests.post(API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def parse_json(text):
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
            continue

        content = await file.read()

        if i == 0:
            try:
                text = call_gemini(VEHICLE_PROMPT, content, file.content_type)
                vehicle_info = parse_json(text)
            except Exception:
                vehicle_info = {"marque": "Inconnu", "modele": "Inconnu", "couleur": "Inconnu", "type": "Inconnu"}
            time.sleep(5)

        try:
            text = call_gemini(INSPECTION_PROMPT, content, file.content_type)
            result = parse_json(text)
            for d in result.get("defauts", []):
                d["photo_source"] = file.filename
                all_defauts.append(d)
            for r in result.get("recommandations", []):
                all_recommandations.add(r)
            scores.append(result.get("score_global", 100))
        except Exception as e:
            errors.append(f"{file.filename}: erreur - {str(e)}")

        time.sleep(5)

    if not scores:
        raise HTTPException(status_code=500, detail=f"Erreurs: {errors}")

    score_final = round(sum(scores) / len(scores))
    cout_min = sum(d.get("cout_reparation_min", 0) for d in all_defauts)
    cout_max = sum(d.get("cout_reparation_max", 0) for d in all_defauts)
    all_defauts.sort(key=lambda x: {"grave": 0, "modere": 1, "mineur": 2}.get(x.get("severite", "mineur"), 3))

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
