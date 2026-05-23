from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import json
from typing import Dict, Any

# Initialize the API
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

print("Loading ML Model and Feature Map...")
# Load the saved model
model = xgb.XGBClassifier()
model.load_model("churn_model.json")

# Load the expected columns so our API knows exactly what shape the data should be
with open("expected_features.json", "r") as f:
    expected_features = json.load(f)

# Define how the incoming data should look
class CustomerData(BaseModel):
    data: Dict[str, Any]

@app.post("/predict")
def predict_churn(payload: CustomerData):
    try:
        # Convert incoming JSON data into a pandas DataFrame
        df = pd.DataFrame([payload.data])
        
        # Apply the exact same preprocessing (One-Hot Encoding) as our training script
        df_encoded = pd.get_dummies(df)
        
        # Align the columns: Add missing columns as 0, and drop extra ones
        for col in expected_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        # Ensure the columns are in the exact same order as training
        df_encoded = df_encoded[expected_features]
        
        # Run the prediction
        prediction = model.predict(df_encoded)[0]
        probability = model.predict_proba(df_encoded)[0][1]
        
        # Return the results as JSON
        return {
            "churn_prediction": int(prediction), # 1 for Yes, 0 for No
            "churn_probability": round(float(probability) * 100, 2) # e.g., 85.5%
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))