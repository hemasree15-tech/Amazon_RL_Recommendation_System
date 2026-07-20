"""
q_learning_model.py
---------------------
Reinforcement Learning Recommendation System using Q-Learning.

Concept:
--------
Agent       -> The recommendation system itself
Environment -> The Amazon product interaction data
State       -> Which category the user is currently interested in
Action      -> Which product to recommend from that category
Reward      -> Based on the rating the user gave (if they engage)

Q-Learning Formula:
--------------------
Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]

Where:
  alpha (learning rate)   -> how much new information overrides old
  gamma (discount factor) -> how much future reward matters
  R                       -> reward received for action taken
  max(Q(s',a'))           -> best possible future reward from next state
"""

import pandas as pd
import numpy as np
import random
import pickle

# Reward mapping based on rating
REWARD_MAP = {5: 10, 4: 5, 3: 1, 2: -5, 1: -10}


def prepare_rl_data(df):
    """
    Adds a reward column to the dataframe based on rating.
    """
    df = df.copy()
    df['reward'] = df['rating'].map(REWARD_MAP)
    return df


def build_states_actions(df):
    """
    State = category (user's area of interest)
    Action = product_id within that category
    """
    states = df['category'].unique().tolist()
    actions_by_state = {}

    for state in states:
        products_in_state = df[df['category'] == state]['product_id'].unique().tolist()
        actions_by_state[state] = products_in_state

    return states, actions_by_state


def initialize_q_table(states, actions_by_state):
    """
    Creates a Q-table as a dictionary:
    q_table[state][action] = 0.0 initially
    """
    q_table = {}
    for state in states:
        q_table[state] = {action: 0.0 for action in actions_by_state[state]}
    return q_table


def get_reward(df, category, product_id):
    """
    Returns average reward for a given (state, action) pair
    based on historical ratings in the dataset.
    """
    subset = df[(df['category'] == category) & (df['product_id'] == product_id)]
    if len(subset) == 0:
        return 0
    return subset['reward'].mean()


def train_q_learning(df, states, actions_by_state,
                      episodes=500, alpha=0.1, gamma=0.9,
                      epsilon=0.2):
    """
    Trains the Q-table using the Q-learning update rule.

    alpha   = learning rate
    gamma   = discount factor (importance of future reward)
    epsilon = exploration rate (chance to try random action)
    """
    q_table = initialize_q_table(states, actions_by_state)
    reward_history = []

    for episode in range(episodes):
        state = random.choice(states)
        total_episode_reward = 0

        # simulate a short sequence of 5 recommendation steps per episode
        for step in range(5):
            actions = actions_by_state[state]
            if len(actions) == 0:
                continue

            # epsilon-greedy policy: explore vs exploit
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)          # explore
            else:
                action = max(q_table[state], key=q_table[state].get)  # exploit best known action

            reward = get_reward(df, state, action)

            next_state = random.choice(states)  # user may move to a new interest/category
            next_actions = actions_by_state[next_state]

            if len(next_actions) > 0:
                max_next_q = max(q_table[next_state].values())
            else:
                max_next_q = 0

            # ---- Q-LEARNING UPDATE FORMULA ----
            old_value = q_table[state][action]
            q_table[state][action] = old_value + alpha * (reward + gamma * max_next_q - old_value)

            total_episode_reward += reward
            state = next_state

        reward_history.append(total_episode_reward)

    return q_table, reward_history


def recommend_for_user_rl(user_id, df, q_table, top_n=5):
    """
    Uses the trained Q-table to recommend top products
    for the user's most engaged category.
    """
    user_data = df[df['user_id'] == user_id]

    if len(user_data) == 0:
        favourite_category = random.choice(list(q_table.keys()))
    else:
        favourite_category = user_data['category'].mode()[0]

    action_values = q_table.get(favourite_category, {})
    if not action_values:
        return pd.DataFrame(columns=['product_id', 'product_name', 'q_value'])

    sorted_actions = sorted(action_values.items(), key=lambda x: x[1], reverse=True)[:top_n]

    result = pd.DataFrame(sorted_actions, columns=['product_id', 'q_value'])
    name_map = df.drop_duplicates('product_id').set_index('product_id')['product_name']
    result['product_name'] = result['product_id'].map(name_map)

    return result[['product_id', 'product_name', 'q_value']]


def save_q_table(q_table, path="models/q_table.pkl"):
    with open(path, "wb") as f:
        pickle.dump(q_table, f)
    print("Q-table saved to", path)


if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data

    data = load_data("../dataset/amazon_reviews.csv")
    data = clean_data(data)
    data = prepare_rl_data(data)

    states, actions_by_state = build_states_actions(data)

    print("Training Q-Learning agent...")
    q_table, reward_history = train_q_learning(data, states, actions_by_state, episodes=500)

    print("\nTraining completed.")
    print("Average reward (last 50 episodes):", np.mean(reward_history[-50:]))

    save_q_table(q_table, "../models/q_table.pkl")

    sample_user = data['user_id'].iloc[0]
    print("\nRL Recommendations for user:", sample_user)
    print(recommend_for_user_rl(sample_user, data, q_table, top_n=5))
