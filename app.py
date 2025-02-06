from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import pandas as pd
from io import StringIO
import io

app = FastAPI()

torch.serialization.add_safe_globals(["TreeClassificator2LayersNN"])

model = torch.load("best_model.pth", map_location=torch.device('cpu'), weights_only=False)
model.eval()


class InputData(BaseModel):
    data: list

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    data = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    input_data = torch.tensor(data.values, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(input_data)

    predictions_list = predictions.numpy().tolist()
    
    return {"predictions": predictions_list}