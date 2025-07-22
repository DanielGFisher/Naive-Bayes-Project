class DataProcessor:
    def prepare_data(df, label_column):
        X = df.drop(columns=[label_column]).values
        y = df[label_column].values
        return X, y

    def split_data(self, X, y, ratio=0.7):
        split_index = int(len(X) * ratio)
        return X[:split_index], y[:split_index], X[split_index:], y[split_index:]
