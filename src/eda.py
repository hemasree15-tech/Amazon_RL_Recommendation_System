"""
eda.py
------
Exploratory Data Analysis for Amazon Reviews Dataset.
Generates graphs and saves them to the graphs/ folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def dataset_overview(df):
    print("Total Reviews:", len(df))
    print("Unique Users:", df['user_id'].nunique())
    print("Unique Products:", df['product_id'].nunique())
    print("Categories:", df['category'].unique())


def plot_rating_distribution(df, save_path="graphs/rating_distribution.png"):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='rating', data=df, palette='viridis')
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Reviews")
    plt.savefig(save_path)
    plt.show()


def plot_top_rated_products(df, save_path="graphs/top_rated_products.png"):
    top_products = df.groupby('product_name')['rating'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    top_products.plot(kind='barh', color='teal')
    plt.title("Top 10 Rated Products")
    plt.xlabel("Average Rating")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_most_active_users(df, save_path="graphs/active_users.png"):
    active_users = df['user_id'].value_counts().head(10)
    plt.figure(figsize=(8, 5))
    active_users.plot(kind='bar', color='orange')
    plt.title("Top 10 Most Active Users")
    plt.xlabel("User ID")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_product_popularity(df, save_path="graphs/product_popularity.png"):
    popularity = df['product_id'].value_counts().head(10)
    plt.figure(figsize=(8, 5))
    popularity.plot(kind='bar', color='purple')
    plt.title("Top 10 Most Popular Products")
    plt.xlabel("Product ID")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_reviews_per_category(df, save_path="graphs/reviews_per_category.png"):
    plt.figure(figsize=(8, 5))
    sns.countplot(y='category', data=df, order=df['category'].value_counts().index, palette='mako')
    plt.title("Number of Reviews per Category")
    plt.xlabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_reward_distribution(df, save_path="graphs/reward_distribution.png"):
    # reward mapping same as used in Q-learning model
    reward_map = {5: 10, 4: 5, 3: 1, 2: -5, 1: -10}
    df['reward'] = df['rating'].map(reward_map)

    plt.figure(figsize=(6, 4))
    sns.countplot(x='reward', data=df, palette='coolwarm')
    plt.title("Reward Distribution (based on Ratings)")
    plt.xlabel("Reward Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_correlation(df, save_path="graphs/correlation.png"):
    temp = df.copy()
    temp['review_length'] = temp['review_text'].apply(len)
    numeric_df = temp[['rating', 'review_length']]

    plt.figure(figsize=(5, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues')
    plt.title("Correlation: Rating vs Review Length")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data

    data = load_data("../dataset/amazon_reviews.csv")
    data = clean_data(data)

    dataset_overview(data)
    plot_rating_distribution(data)
    plot_top_rated_products(data)
    plot_most_active_users(data)
    plot_product_popularity(data)
    plot_reviews_per_category(data)
    plot_reward_distribution(data)
    plot_correlation(data)
