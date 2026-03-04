from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from sklearn.exceptions import NotFittedError
from model import KeystrokeModel
from utils import extract_features

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

model = KeystrokeModel()
model.load()

training_data = []

class KeystrokeInput(BaseModel):
    keystrokes: list

@app.post("/train")
def train(data: KeystrokeInput):
    features = extract_features(data.keystrokes)
    if features.size == 0:
        return {"status": "Nenhum dado de digitação capturado."}

    training_data.append(features)

    if len(training_data) >= 5:
        X = np.stack(training_data)
        model.train(X)
        training_data.clear()
        return {"status": "Modelo treinado"}

    return {"status": f"Amostras coletadas: {len(training_data)}"}

@app.post("/verify")
def verify(data: KeystrokeInput):
    features = extract_features(data.keystrokes)
    if features.size == 0:
        return {"result": "Nenhum dado de digitação capturado."}

    try:
        prediction = model.predict([features])
    except NotFittedError:
        return {"result": "Modelo ainda não foi treinado. Clique em Treinar algumas vezes antes de verificar."}

    if prediction[0] == 1:
        return {"result": "Usuário legítimo"}
    else:
        return {"result": "Intruso detectado"}
