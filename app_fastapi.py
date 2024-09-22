from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import pandas as pd
import pickle
from typing import Optional
from pycaret.regression import predict_model

path2 = "D:/Users/Asus\Documents/PYTHON PARA APIS/Taller 1 (2 corte)/"

# Crear una instancia de FastAPI
app = FastAPI()

# Definir el archivo JSON donde se guardarán las predicciones
file_name = 'predicciones.json'

# Cargar el modelo preentrenado desde el archivo pickle
model_path = "best_model.pkl"
with open(path2 + 'best_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

# Cargar base de predicción en kaggle
prueba = pd.read_csv(path2 + "prueba_APP.csv",header = 0,sep=";",decimal=",")
prueba.drop(columns=['Address','price'], inplace=True)

## Datos de Entrada
dominio =  'yahoo'
Tec = 'PC'
Avg = 33.946241
Time_App = 10.983977
Time_Web = 37.951489
Length = 3.050713

# Modelo de datos para la API (simplificado, adaptado según tus variables)
class InputData(BaseModel):
    Email: str 
    dominio: str
    Tec: str
    Avg: float
    Time_App: float
    Time_Web: float
    Length : float 


@app.get("/")
def home():
    # Retorna un simple mensaje de texto
    return 'Predicción precio'

# Función para guardar predicciones en un archivo JSON
def save_prediction(prediction_data):
    try:
        with open(file_name, 'r') as file:
            predictions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions = []

    predictions.append(prediction_data)

    with open(file_name, 'w') as file:
        json.dump(predictions, file, indent=4)

# Endpoint para realizar la predicción
@app.post("/predict")
def predict(data: InputData):
    # Crear DataFrame a partir de los datos de entrada
    user_data = pd.DataFrame([data.dict()])
    # Asegurar que las columnas del DataFrame "user_data" coincidan con las de "prueba"
    user_data.columns = prueba.columns
    # Concatenar los datos del usuario con los datos del CSV "prueba"
    prueba2 = pd.concat([user_data, prueba], axis=0)
    prueba2.index = range(prueba2.shape[0])
    
    # Realizar predicción
    predictions = predict_model(modelo, data=prueba2)
    predictions["price"] = predictions["prediction_label"]
    prediction_label = predictions.iloc[0,]["price"]

    # Guardar predicción con ID en el archivo JSON
    prediction_result = {"Email": data.Email, "prediction": prediction_label}
    save_prediction(prediction_result)
    
    return prediction_result

# Ejecutar la aplicación si se llama desde la terminal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)