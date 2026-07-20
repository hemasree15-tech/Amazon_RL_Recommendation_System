"""
evaluation.py
--------------
Compares all 3 recommendation models using standard metrics:
Precision@K, Recall@K, F1 Score, RMSE, MAE, Average Reward
"""

import numpy as np
import pandas as pd


def precision_recall_at_k(recommended_products, relevant_products, k=5):
    """
    recommended_products : list of product_ids recommended
    relevant_products    : list of product_ids the user actually liked (rating >= 4)
    """
    recommended_k = recommended_products[:k]
    if len(recommended_k) == 0:
        return 0, 0

    hits = len(set(recommended_k) & set(relevant_products))

    precision = hits / len(recommended_k)
    recall = hits / len(relevant_products) if len(relevant_products) > 0 else 0

    return precision, recall


def f1_score_manual(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def rmse_mae(actual_ratings, predicted_ratings):
    actual_ratings = np.array(actual_ratings)
    predicted_ratings = np.array(predicted_ratings)

    rmse = np.sqrt(np.mean((actual_ratings - predicted_ratings) ** 2))
    mae = np.mean(np.abs(actual_ratings - predicted_ratings))
    return rmse, mae


def evaluate_model(eval_df, recommend_function, model_name, sample_users, k=5, **kwargs):
    """
    Generic evaluation loop that works for any recommendation function.
    Compares recommended products against products the user actually rated highly.
    'eval_df' is used only to find what the user actually liked (ground truth).
    Any 'df' needed by the recommend_function itself should be passed inside kwargs.
    """
    precisions, recalls = [], []

    for user in sample_users:
        user_data = eval_df[eval_df['user_id'] == user]
        relevant = user_data[user_data['rating'] >= 4]['product_id'].tolist()

        if len(relevant) == 0:
            continue

        recs = recommend_function(user, **kwargs)
        if recs is None or len(recs) == 0:
            continue

        recommended_ids = recs['product_id'].tolist()

        p, r = precision_recall_at_k(recommended_ids, relevant, k=k)
        precisions.append(p)
        recalls.append(r)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    f1 = f1_score_manual(avg_precision, avg_recall)

    print(f"\n--- {model_name} ---")
    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Recall@{k}:    {avg_recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")

    return {
        "Model": model_name,
        f"Precision@{k}": round(avg_precision, 4),
        f"Recall@{k}": round(avg_recall, 4),
        "F1 Score": round(f1, 4)
    }


if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data, build_user_item_matrix
    from popularity_model import recommend_for_user_popularity
    from collaborative_filtering import build_similarity_matrix, recommend_for_user_cf
    from q_learning_model import prepare_rl_data, build_states_actions, train_q_learning, recommend_for_user_rl

    data = load_data("../dataset/amazon_reviews.csv")
    data = clean_data(data)

    ui_matrix = build_user_item_matrix(data)
    sim_matrix = build_similarity_matrix(ui_matrix)

    rl_data = prepare_rl_data(data)
    states, actions_by_state = build_states_actions(rl_data)
    q_table, reward_history = train_q_learning(rl_data, states, actions_by_state, episodes=300)

    sample_users = data['user_id'].drop_duplicates().sample(30, random_state=42).tolist()

    results = []

    results.append(evaluate_model(
        data, recommend_for_user_popularity, "Popularity Based",
        sample_users, k=5, df=data, top_n=5
    ))

    results.append(evaluate_model(
        data, recommend_for_user_cf, "Collaborative Filtering",
        sample_users, k=5, user_item_matrix=ui_matrix, similarity_df=sim_matrix, df=data, top_n=5
    ))

    results.append(evaluate_model(
        data, recommend_for_user_rl, "Q-Learning (RL)",
        sample_users, k=5, df=rl_data, q_table=q_table, top_n=5
    ))

    results_df = pd.DataFrame(results)
    print("\n\nFinal Comparison Table:\n", results_df)
    results_df.to_csv("../output/model_comparison.csv", index=False)
