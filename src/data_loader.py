"""
# data_loader.py
# Task: 1 - EDA & Preprocessing
# Created by: Addisu Taye
# Date: July 4, 2025
# Purpose: Load and filter complaint dataset based on product type and non-empty narratives
# Key Features: Loads raw data, filters by product, removes empty narratives
"""

import pandas as pd
from config import DATA_PATH, FILTERED_DATA_PATH, PRODUCTS_TO_KEEP

def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("✅ Raw data loaded")
    return df

def filter_data(df):
    filtered_df = df[df['Product'].isin(PRODUCTS_TO_KEEP)]
    filtered_df = filtered_df[filtered_df['Consumer complaint narrative'].notnull()]
    print("✅ Data filtered by product and non-empty narratives")
    return filtered_df

def save_filtered_data(df):
    df.to_csv(FILTERED_DATA_PATH, index=False)
    print(f"✅ Filtered data saved to {FILTERED_DATA_PATH}")