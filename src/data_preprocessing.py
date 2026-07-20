"""
data_preprocessing.py
----------------------
This file loads the Amazon reviews dataset and cleans it.
Steps:
1. Load CSV file
2. Check dataset info
3. Handle missing values
4. Remove duplicates
5. Prepare user-item interaction table
"""

import pandas as pd

def load_data(path="dataset/amazon_reviews.csv"):
    df = pd.read_csv(path)
    return df


def check_data_info(df):
    print("Shape of dataset:", df.shape)
    print("\nColumn names:", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())


def clean_data(df):
    # remove missing values
    df = df.dropna()

    # remove duplicate rows
    df = df.drop_duplicates()

    # make sure rating is numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])

    # keep rating between 1 and 5 only
    df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]

    # convert timestamp to date format
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    df = df.reset_index(drop=True)
    return df


def build_user_item_matrix(df):
    """
    Creates a user-item interaction matrix.
    Rows = users, Columns = products, Values = rating
    """
    matrix = df.pivot_table(index='user_id', columns='product_id',
                             values='rating', aggfunc='mean')
    matrix = matrix.fillna(0)
    return matrix


# Test this file independently
if __name__ == "__main__":
    data = load_data("../dataset/amazon_reviews.csv")
    check_data_info(data)
    data = clean_data(data)
    print("\nShape after cleaning:", data.shape)

    ui_matrix = build_user_item_matrix(data)
    print("\nUser-Item matrix shape:", ui_matrix.shape)
