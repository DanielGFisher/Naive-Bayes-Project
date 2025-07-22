from xml.etree.ElementPath import prepare_predicate

from fastapi import FastAPI, Request
import pandas as pd
from Models.Classifier import NaiveBayesPredictor
from Models.Data_Processor import DataProcessor
from Models.Probability_Store import ProbabilityStore
from Models.Trainer import NaiveBayesTrainer
from Models.Validator import Validator

app = FastAPI()

store = ProbabilityStore()
trainer = NaiveBayesTrainer(store)
predictor = NaiveBayesPredictor(store)
processor = DataProcessor()

@app.post("/training/")
def train_model(file_path: str = r"C:\Users\danie\Downloads\phishing.csv", label_column: str = "class"):
    df = pd.read_csv(file_path)
    X, y = processor.prepare_data(df, label_column)
    X_train, y_train, _, _ = processor.split_data(X, y)
    trainer.fit(X_train, y_train)
    return {"message" : "Training Complete"}

@app.post("/predict")
def predict(input_data: dict):
    features = list(input_data.values())
    prediction = predictor.predict([features])
    return {"prediction" : prediction[0]}

@app.post("/validate")
def validate(request: Request):
    pass

