import pandas as pd
import pickle

model_path = "./static/models/RandomForest.pkl"
with open(model_path, "rb") as f:
  model = pickle.load(f)

def predict_crop(features):
  prediction = model.predict(features)[0]
  print(prediction)
  return prediction