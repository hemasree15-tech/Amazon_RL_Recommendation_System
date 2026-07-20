"""
popularity_model.py
--------------------
Baseline Model 1: Popularity-Based Recommendation
Recommends the same top-rated products to every user,
based on average rating and number of reviews.
"""

import pandas as pd


def popularity_recommendation(df, top_n=5, min_reviews=5):
    """
    Ranks products by average rating (only products with
    at least 'min_reviews' reviews, to avoid bias from single reviews).
    """
    stats = df.groupby('product_id').agg(
        avg_rating=('rating', 'mean'),
        num_reviews=('rating', 'count'),
        product_name=('product_name', 'first')
    ).reset_index()

    stats = stats[stats['num_reviews'] >= min_reviews]
    stats = stats.sort_values(by=['avg_rating', 'num_reviews'], ascending=False)

    top_products = stats.head(top_n)
    return top_products[['product_id', 'product_name', 'avg_rating', 'num_reviews']]


def recommend_for_user_popularity(user_id, df, top_n=5):
    """
    Popularity model gives same recommendations to all users
    (it does not personalize). 'user_id' is accepted only to keep
    the same function signature as the other recommendation models.
    """
    return popularity_recommendation(df, top_n=top_n)


if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data

    data = load_data("../dataset/amazon_reviews.csv")
    data = clean_data(data)

    print("Top 5 Popular Products:\n")
    print(popularity_recommendation(data, top_n=5))

    sample_user = data['user_id'].iloc[0]
    print("\nSame recommendations shown to sample user", sample_user, ":\n")
    print(recommend_for_user_popularity(sample_user, data, top_n=5))
