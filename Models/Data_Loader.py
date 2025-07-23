import pandas as pd

class CSVLoader:
    def load_csv(self, file_path):
        return pd.read_csv(file_path)

