from Models.Data_Loader import CSVLoader
from Models.Data_Processor import DataProcessor
from Models.Trainer import NaiveBayesTrainer
from Models.Probability_Store import ProbabilityStore
from Models.Predictor import NaiveBayesPredictor
from Models.Validator import Validator


def main():
    print("--- Naive Bayes Categorical Classifier ---\n")

    # 1 - Load CSV
    loader = CSVLoader()
    file_path = r"C:\Users\danie\Downloads\phishing.csv"
    label_column = "class"
    df = loader.load_csv(file_path)

    # 2 - Prepare Data
    processor = DataProcessor()
    X, y = processor.prepare_data(df, label_column)
    X_train, y_train, X_test, y_test = processor.split_data(X, y, ratio=0.7)

    # 3 - Train
    print("Training model...")
    store = ProbabilityStore()
    trainer = NaiveBayesTrainer(store)
    trainer.fit(X_train, y_train)

    # 4 - Predict
    print("Predicting test set...")
    predictor = NaiveBayesPredictor(store)
    predictions = predictor.predict(X_test)

    # 5 - Validate
    print("\nValidation Report:")
    validator = Validator(y_test, predictions)
    validator.print_report()


if __name__ == "__main__":
    main()
