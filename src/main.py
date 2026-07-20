"""
main.py
--------
RUN THIS FILE IN SPYDER.

Personalized Product Recommendation System Using Reinforcement Learning
Based on Amazon Product Reviews.

This file runs the complete pipeline:
1. Load and clean data
2. EDA (graphs)
3. Popularity-based recommendation (baseline)
4. Collaborative filtering (baseline)
5. Q-Learning recommendation (main RL model)
6. Evaluate and compare all 3 models
7. Generate final recommendations and save to CSV
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Make sure the script works no matter what folder Spyder is
# currently pointed at. We build paths relative to THIS file's
# location, then switch into the project root folder.
# ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                # project root
import sys
sys.path.insert(0, SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

# make sure output folders exist
os.makedirs("graphs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

from data_preprocessing import load_data, check_data_info, clean_data, build_user_item_matrix
from eda import (dataset_overview, plot_rating_distribution, plot_top_rated_products,
                  plot_most_active_users, plot_product_popularity,
                  plot_reviews_per_category, plot_reward_distribution, plot_correlation)
from popularity_model import recommend_for_user_popularity
from collaborative_filtering import build_similarity_matrix, recommend_for_user_cf
from q_learning_model import (prepare_rl_data, build_states_actions,
                               train_q_learning, recommend_for_user_rl, save_q_table)
from evaluation import evaluate_model
from visualization import plot_model_comparison, plot_reward_progress, plot_cumulative_reward


# =========================================================
# STEP 1: LOAD AND CLEAN DATA
# =========================================================
print("STEP 1: Loading dataset...")
df = load_data("dataset/amazon_reviews.csv")
check_data_info(df)

df = clean_data(df)
print("\nShape after cleaning:", df.shape)


# =========================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# =========================================================
print("\nSTEP 2: Running EDA...")
dataset_overview(df)
plot_rating_distribution(df)
plot_top_rated_products(df)
plot_most_active_users(df)
plot_product_popularity(df)
plot_reviews_per_category(df)
plot_reward_distribution(df)
plot_correlation(df)


# =========================================================
# STEP 3: USER-ITEM MATRIX (needed for collaborative filtering)
# =========================================================
print("\nSTEP 3: Building user-item matrix...")
ui_matrix = build_user_item_matrix(df)
sim_matrix = build_similarity_matrix(ui_matrix)
print("User-Item matrix shape:", ui_matrix.shape)


# =========================================================
# STEP 4: TRAIN Q-LEARNING MODEL (REINFORCEMENT LEARNING)
# =========================================================
print("\nSTEP 4: Training Q-Learning agent...")
rl_data = prepare_rl_data(df)
states, actions_by_state = build_states_actions(rl_data)

q_table, reward_history = train_q_learning(
    rl_data, states, actions_by_state,
    episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2
)

print("Average reward (last 50 episodes):", np.mean(reward_history[-50:]))
save_q_table(q_table, "models/q_table.pkl")

plot_reward_progress(reward_history)
plot_cumulative_reward(reward_history)


# =========================================================
# STEP 5: EVALUATE ALL 3 MODELS
# =========================================================
print("\nSTEP 5: Evaluating models...")
sample_users = df['user_id'].drop_duplicates().sample(30, random_state=42).tolist()

results = []

results.append(evaluate_model(
    df, recommend_for_user_popularity, "Popularity Based",
    sample_users, k=5, df=df, top_n=5
))

results.append(evaluate_model(
    df, recommend_for_user_cf, "Collaborative Filtering",
    sample_users, k=5, user_item_matrix=ui_matrix, similarity_df=sim_matrix, df=df, top_n=5
))

results.append(evaluate_model(
    df, recommend_for_user_rl, "Q-Learning (RL)",
    sample_users, k=5, df=rl_data, q_table=q_table, top_n=5
))

results_df = pd.DataFrame(results)
print("\nFinal Model Comparison:\n", results_df)

plot_model_comparison(results_df)
results_df.to_csv("output/model_comparison.csv", index=False)


# =========================================================
# STEP 6: GENERATE FINAL RECOMMENDATIONS FOR SAMPLE USERS
# =========================================================
print("\nSTEP 6: Generating final recommendations...")

final_recommendations = []

for user in sample_users[:10]:
    recs = recommend_for_user_rl(user, rl_data, q_table, top_n=5)
    for _, row in recs.iterrows():
        final_recommendations.append({
            "user_id": user,
            "product_id": row['product_id'],
            "product_name": row['product_name'],
            "q_value": row['q_value']
        })

recommendations_df = pd.DataFrame(final_recommendations)
recommendations_df.to_csv("output/recommendations.csv", index=False)

print("\nSample Recommendations:\n")
print(recommendations_df.head(15))

print("\n===================================================")
print("PROJECT PIPELINE COMPLETED SUCCESSFULLY")
print("Check the 'graphs/' folder for visualizations")
print("Check the 'output/' folder for CSV results")
print("===================================================")
