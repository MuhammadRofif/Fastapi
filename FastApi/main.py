from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="API Prediksi Makanan dan Minuman Sehat vs Tidak Sehat")

# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Skema input untuk data makanan dan minuman
class Food(BaseModel):
    name: str
    calories: float
    proteins: float
    fat: float
    carbohydrate: float

# Fungsi untuk memproses input
def preprocess_input(data: Food):
    # Membuat DataFrame dari input data
    df = pd.DataFrame([{
        "calories": data.calories,
        "proteins": data.proteins,
        "fat": data.fat,
        "carbohydrate": data.carbohydrate
    }])

    # Menambahkan kolom total_macros (total gizi makro)
    df['total_macros'] = df['proteins'] + df['fat'] + df['carbohydrate']

    # Normalisasi data hanya pada fitur numerik yang dibutuhkan
    df_scaled = scaler.transform(df[['calories', 'proteins', 'fat', 'carbohydrate', 'total_macros']])
    return df_scaled

@app.get("/")
def read_root():
    return {"message": "API Prediksi Makanan Sehat vs Tidak Sehat sedang berjalan"}

# Endpoint untuk prediksi makanan sehat atau tidak sehat
@app.post("/predict")
def predict_health_status(data: Food):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    result = "Sehat" if prediction == 1 else "Tidak Sehat"
    return {
        "name": data.name, 
        "prediction": int(prediction),
        "result": result,
    }
