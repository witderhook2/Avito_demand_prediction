import pickle
from typing import Optional
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Avito Deal Probability API")

with open("catboost_model.pkl", "rb") as f:
  model = pickle.load(f)

request_count = 0


def gradient_text_preprocessing_transform(df_valid: pd.DataFrame) -> pd.DataFrame:
  df = df_valid.copy()

  df["activation_date"] = pd.to_datetime(df["activation_date"])
  df["activation_day_of_week"] = df["activation_date"].dt.day_of_week.astype(str)
  df["image_top_1"] = df["image_top_1"].astype(str)
  df["title"] = df["title"].astype(str)
  df["description"] = df["description"].astype(str)

  cat_features = [
      "region",
      "city",
      "parent_category_name",
      "category_name",
      "param_1",
      "activation_day_of_week",
      "user_type",
      "image_top_1",
  ]
  for col in cat_features:
    df[col] = df[col].astype(str)

  feature_names = [
      "region",
      "city",
      "parent_category_name",
      "category_name",
      "param_1",
      "activation_day_of_week",
      "user_type",
      "image_top_1",
      "price",
      "item_seq_number",
      "title",
      "description",
  ]

  return df[feature_names]


class AvitoItemInput(BaseModel):
  region: str
  city: str
  parent_category_name: str
  category_name: str
  param_1: Optional[str] = ""
  activation_date: str
  user_type: str
  image_top_1: Optional[str] = "nan"
  price: Optional[float] = None
  item_seq_number: int
  title: str
  description: Optional[str] = ""


@app.get("/stats")
def stats():
  return {"request_count": request_count}


@app.get("/health")
def health():
  return {"status": "OK"}


@app.post("/predict_model")
def predict_model(input_data: AvitoItemInput):
  global request_count
  request_count += 1

  try:
    raw_df = pd.DataFrame([input_data.model_dump()])

    X_processed = gradient_text_preprocessing_transform(raw_df)

    prediction = model.predict(X_processed)
    deal_prob = float(np.clip(prediction[0], 0, 1))

    return {"deal_probability": round(deal_prob, 5)}

  except Exception as e:
    raise HTTPException(
        status_code=400, detail=f"Ошибка при обработке данных: {str(e)}"
    )


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=5000)
