import uvicorn
from fastapi import FastAPI
import pandas as pd
from Models.Data_Processor import DataProcessor
from Models.Probability_Store import ProbabilityStore
from Models.Trainer import NaiveBayesTrainer
from Models.Validator import Validator
from Models.Classifier import NaiveBayesPredictor


app = FastAPI()

store = ProbabilityStore()
trainer = NaiveBayesTrainer(store)
predictor = NaiveBayesPredictor(store)
processor = DataProcessor()

@app.post("/training/")
def train_model(file_path: str = "data.csv", label_column: str = "class"):
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
def validate_model(file_path: str = "data.csv", label_column: str = "class"):
        df = pd.read_csv(file_path)
        X, y = processor.prepare_data(df, label_column)
        X_train, y_train, X_test, y_test = processor.split_data(X, y)

        trainer.fit(X_train, y_train)
        y_pred = predictor.predict(X_test)

        validator = Validator(y_test, y_pred)
        return {
            "TP": validator.TP,
            "TN": validator.TN,
            "FP": validator.FP,
            "FN": validator.FN,
            "Accuracy": round(validator.accuracy(), 2),
            "Precision": round(validator.precision(), 2),
            "Recall": round(validator.recall(), 2),
            "F1 Score": round(validator.f1_score(), 2)
        }

if __name__ == "__main__":
    uvicorn.run("Fast_API:app", host="0.0.0.0", port=8000)