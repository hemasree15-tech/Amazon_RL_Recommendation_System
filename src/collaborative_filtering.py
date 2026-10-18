"""
collaborative_filtering.py
----------------------------
Baseline Model 2: User-Based Collaborative Filtering
Finds users similar to the target user, and recommends
products liked by those similar users.

Note: this dataset has ~9,000 users, so instead of building a huge
users x users similarity table, we store the normalised user vectors
once and calculate similarity only for the user we need.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def build_similarity_matrix(user_item_matrix):
    """
    Prepares the user vectors for cosine similarity.
    After normalising every row to length 1, the cosine similarity
    between two users is simply the dot product of their rows.
    """
    return normalize(csr_matrix(user_item_matrix.to_numpy()))


def recommend_for_user_cf(user_id, user_item_matrix, user_vectors, df, top_n=5, k_neighbors=10):
    """
    Steps:
    1. Find top-k most similar users to the given user
    2. Look at products those similar users rated highly
    3. Remove products the target user already rated
    4. Return top-N products
    """
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=['product_id', 'product_name', 'score'])

    user_pos = user_item_matrix.index.get_loc(user_id)
    ratings = user_item_matrix.to_numpy()

    # cosine similarity of this user with every other user
    similarity = user_vectors[user_pos].dot(user_vectors.T).toarray().ravel()
    similarity[user_pos] = -1          # exclude the user themselves

    # top-k most similar users (only those with some similarity)
    neighbors = np.argsort(similarity)[::-1][:k_neighbors]
    neighbors = [n for n in neighbors if similarity[n] > 0]
    if not neighbors:
        return pd.DataFrame(columns=['product_id', 'product_name', 'score'])

    # weighted score using similarity of neighbors
    scores = similarity[neighbors] @ ratings[neighbors]

    # remove products already rated by the target user
    scores[ratings[user_pos] > 0] = 0

    top_idx = np.argsort(scores)[::-1][:top_n]
    top_idx = [i for i in top_idx if scores[i] > 0]
    if not top_idx:
        return pd.DataFrame(columns=['product_id', 'product_name', 'score'])

    result = pd.DataFrame({
        'product_id': user_item_matrix.columns[top_idx],
        'score': scores[top_idx]
    })

    # attach product names
    name_map = df.drop_duplicates('product_id').set_index('product_id')['product_name']
    result['product_name'] = result['product_id'].map(name_map)

    return result[['product_id', 'product_name', 'score']]


if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data, build_user_item_matrix

    data = load_data("../dataset/amazon.csv")
    data = clean_data(data)

    ui_matrix = build_user_item_matrix(data)
    user_vectors = build_similarity_matrix(ui_matrix)

    sample_user = ui_matrix.index[0]
    print("Recommendations for user:", sample_user)
    print(recommend_for_user_cf(sample_user, ui_matrix, user_vectors, data, top_n=5))
