import pandas as pd

class DataCleaner:
    def clean(self, df: pd.DataFrame):
        cleaned_df = df.copy()

        for col in cleaned_df.select_dtypes(include="object").columns:
            mode = cleaned_df[col].mode(dropna=True)
            if not mode.empty:
                cleaned_df[col].fillna(mode[0], inplace=True)
            else:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()

        for col in cleaned_df.select_dtypes(include="number").columns:
            median = cleaned_df[col].median()
            cleaned_df[col].fillna(median, inplace=True)

        return cleaned_df