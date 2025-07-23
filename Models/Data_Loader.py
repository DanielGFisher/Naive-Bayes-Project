import pandas as pd
import os

class CSVLoader:
    def load_csv(self, file_path):
        df = pd.read_csv(file_path)

        df.to_csv("data.csv", index=False)

        return pd.read_csv("data.csv")

