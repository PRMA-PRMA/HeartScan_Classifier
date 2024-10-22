# scripts/data_loader.py

import pandas as pd
import re

def load_data(input_file):
    df = pd.read_csv(input_file)
    df.dropna(subset=['Report_Text'], inplace=True)
    return df

def preprocess_text(text):
    # Remove unnecessary characters and normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
