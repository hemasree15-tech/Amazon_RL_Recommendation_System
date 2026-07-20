"""
collaborative_filtering.py
----------------------------
Baseline Model 2: User-Based Collaborative Filtering
Finds users similar to the target user, and recommends
products liked by those similar users.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_matrix(user_item_matrix):
    """
    Calculates cosine similarity between all users
    based on their rating patterns.
    """
    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity,
                                  index=user_item_matrix.index,
                                  columns=user_item_matrix.index)
    return similarity_df


def recommend_for_user_cf(user_id, user_item_matrix, similarity_df, df, top_n=5, k_neighbors=10):
    """
    Steps:
    1. Find top-k most similar users to the given user
    2. Look at products those similar users rated highly
    3. Remove products the target user already rated
    4. Return top-N products
    """
    if user_id not in similarity_df.index:
        return pd.DataFrame(columns=['product_id', 'product_name', 'score'])

    # top similar users (excluding the user themselves)
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:k_neighbors + 1]

    # products already rated by target user
    already_rated = user_item_matrix.loc[user_id]
    already_rated = already_rated[already_rated > 0].index.tolist()

    # weighted score using similarity of neighbors
    scores = {}
    for other_user, sim_score in similar_users.items():
        other_ratings = user_item_matrix.loc[other_user]
        for product_id, rating in other_ratings.items():
            if rating > 0 and product_id not in already_rated:
                scores[product_id] = scores.get(product_id, 0) + sim_score * rating

    if not scores:
        return pd.DataFrame(columns=['product_id', 'product_name', 'score'])

    result = pd.DataFrame(list(scores.items()), columns=['product_id', 'score'])
    result = result.sort_values(by='score', ascending=False).head(top_n)

    # attach product names
    name_map = df.drop_duplicates('product_id').set_index('product_id')['product_name']
    result['product_name'] = result['product_id'].map(name_map)

    return result[['product_id', 'product_name', 'score']]


if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data, build_user_item_matrix

    data = load_data("../dataset/amazon_reviews.csv")
    data = clean_data(data)

    ui_matrix = build_user_item_matrix(data)
    sim_matrix = build_similarity_matrix(ui_matrix)

    sample_user = ui_matrix.index[0]
    print("Recommendations for user:", sample_user)
    print(recommend_for_user_cf(sample_user, ui_matrix, sim_matrix, data, top_n=5))
