"""
visualization.py
------------------
Creates comparison graphs for the 3 models and
graphs the RL agent's learning progress (reward over episodes).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")


def plot_model_comparison(results_df, save_path="graphs/model_comparison.png"):
    metrics = [c for c in results_df.columns if c != "Model"]

    results_df.set_index("Model")[metrics].plot(kind='bar', figsize=(9, 5))
    plt.title("Recommendation Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_reward_progress(reward_history, save_path="graphs/rl_training_progress.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(reward_history, color='green')
    plt.title("Q-Learning Agent: Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_cumulative_reward(reward_history, save_path="graphs/cumulative_reward.png"):
    cumulative = pd.Series(reward_history).cumsum()
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative, color='blue')
    plt.title("Cumulative Reward Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
