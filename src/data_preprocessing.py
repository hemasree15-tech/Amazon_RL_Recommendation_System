"""
data_preprocessing.py
----------------------
This file loads the Amazon products dataset (amazon.csv) and cleans it.

IMPORTANT - how this dataset is organised:
Each row is ONE PRODUCT, and several reviewers of that product are
stored together in a single cell, separated by commas, for example:
    user_id   = "AG3D6O...,AHMY5C...,AHCTC6..."
So the cleaning step "explodes" every row into one row per
(user, product) pair. The dataset only gives the product's AVERAGE
rating (not each reviewer's own rating), so every reviewer of a product
gets that product's rating (rounded to 1-5) as their interaction rating.

Steps:
1. Load CSV file
2. Check dataset info
3. Clean prices / ratings and handle missing values
4. Split the packed reviewer columns into one row per user-product pair
5. Prepare user-item interaction table
"""

import pandas as pd


def load_data(path="dataset/amazon.csv"):
    df = pd.read_csv(path)
    return df


def check_data_info(df):
    print("Shape of dataset:", df.shape)
    print("\nColumn names:", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())


def _to_number(series):
    """Turns text like '₹1,099' or '64%' into a number."""
    cleaned = (series.astype(str)
                     .str.replace("₹", "", regex=False)
                     .str.replace(",", "", regex=False)
                     .str.replace("%", "", regex=False)
                     .str.strip())
    return pd.to_numeric(cleaned, errors='coerce')


def clean_data(df):
    df = df.copy()

    # ---- numeric columns stored as text (₹1,099 / 64% / 24,269) ----
    for col in ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count']:
        df[col] = _to_number(df[col])
    df['rating_count'] = df['rating_count'].fillna(df['rating_count'].median())

    # ---- rating: one row has a typo value, so convert safely ----
    df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace("|", "", regex=False),
                                 errors='coerce')
    df = df.dropna(subset=['rating', 'user_id', 'product_id', 'product_name', 'category'])

    # keep rating between 1 and 5 only
    df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]

    # ---- category looks like "Electronics|HomeTheater|Speakers" ----
    # main category = first part (used as the RL "state"),
    # sub category  = last part
    df['sub_category'] = df['category'].str.split('|').str[-1]
    df['category'] = df['category'].str.split('|').str[0]

    # ---- split the packed reviewer columns (comma separated) ----
    df['user_id'] = df['user_id'].str.split(',')
    df['user_name'] = df['user_name'].str.split(',')

    # only keep rows where ids and names line up one-to-one
    same_length = df['user_id'].str.len() == df['user_name'].str.len()
    df = df[same_length]

    df = df.explode(['user_id', 'user_name'])
    df['user_id'] = df['user_id'].str.strip()

    # the dataset gives the product's average rating (e.g. 4.2), so keep it
    # and also make a whole-number rating (1-5) used by the models
    df = df.rename(columns={'rating': 'avg_rating'})
    df['rating'] = df['avg_rating'].round().clip(1, 5).astype(int)

    # remove duplicate rows (same user reviewing the same product twice)
    df = df.drop_duplicates(subset=['user_id', 'product_id'])

    keep = ['user_id', 'user_name', 'product_id', 'product_name', 'category',
            'sub_category', 'rating', 'avg_rating', 'rating_count',
            'discounted_price', 'actual_price', 'discount_percentage']
    df = df[keep].reset_index(drop=True)
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
    data = load_data("../dataset/amazon.csv")
    check_data_info(data)
    data = clean_data(data)
    print("\nShape after cleaning:", data.shape)

    ui_matrix = build_user_item_matrix(data)
    print("\nUser-Item matrix shape:", ui_matrix.shape)
