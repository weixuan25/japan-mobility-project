import pandas as pd
from collections import Counter
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("loading data...")


mobility_files = ['C:/Users/chow1/Downloads/cityA_groundtruthdata.csv.gz']
mobility_data = pd.concat([pd.read_csv(file).head(40000) for file in mobility_files])

# Combine date and time into a single timestamp column
mobility_data['timestamp'] = pd.to_datetime(mobility_data['d'] + mobility_data['t'])
mobility_data = mobility_data[['uid', 'timestamp', 'x', 'y']]

print("processing...")


# Sort by user ID and timestamp
mobility_data.sort_values(by=['uid', 'timestamp'], inplace=True)

# Generate pairs of consecutive locations
pairs = []
for uid, group in mobility_data.groupby('uid'):
    locations = group[['x', 'y']].values
    for i in range(1, len(locations)):
        pairs.append((tuple(locations[i - 1]), tuple(locations[i])))

# Count the frequency of each pair
pair_counts = Counter(pairs)

# Define minimum support (e.g., 5% of total pairs in this task)
min_support = 0.05 * len(pairs)
frequent_pairs = [pair for pair, count in pair_counts.items() if count >= min_support]


def calculate_user_similarity(mobility_data, recent_n=5):
    """
    Calculate pairwise cosine similarity between users based on the most recent N locations.
    """
    user_locations = {}

    # Group the data by user and collect their most recent N locations
    for uid, group in mobility_data.groupby('uid'):
        locations = group[['x', 'y']].values[-recent_n:]  # Get the last N locations
        user_locations[uid] = locations.flatten()  # Flatten to a 1D vector for similarity calculation

    user_matrix = np.array(list(user_locations.values()))

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(user_matrix)

    user_similarity = {}
    for i, uid1 in enumerate(user_locations.keys()):
        user_similarity[uid1] = {}
        for j, uid2 in enumerate(user_locations.keys()):
            if i != j:
                user_similarity[uid1][uid2] = similarity_matrix[i][j]

    return user_similarity


def predict_top_locations(uid, last_location, pair_counts, user_similarity, top_n=5, similarity_threshold=0.5):
    """
    Predict the top N next locations using user similarity, but only if no historical data is available.
    """
    # First, check if there is enough historical data (i.e., frequent pairs)
    next_locations = {
        pair[1]: count for pair, count in pair_counts.items() if pair[0] == last_location
    }

    # If historical data is available, use it for prediction
    if next_locations:
        return sorted(next_locations.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Otherwise, fall back to user similarity-based prediction
    if uid in user_similarity:
        similar_users = sorted(user_similarity[uid].items(), key=lambda x: x[1], reverse=True)
        
        # Consider only similar users with high similarity (threshold)
        for similar_uid, sim_score in similar_users:
            if sim_score < similarity_threshold:
                break  # No need to consider users with low similarity for this task
            
            # Get the last known location of the similar user
            similar_last_location = mobility_data[mobility_data['uid'] == similar_uid][['x', 'y']].iloc[-1].values
            similar_last_location = tuple(similar_last_location)

            # Get the next locations of this similar user
            similar_next_locations = {
                pair[1]: count for pair, count in pair_counts.items() if pair[0] == similar_last_location
            }

            # Merge the results with the original next locations
            for loc, count in similar_next_locations.items():
                next_locations[loc] = next_locations.get(loc, 0) + count

    # If still no prediction can be made, fall back to global top locations
    if not next_locations:
        # Fallback to the most frequent pairs in the entire dataset
        global_top_locations = pair_counts.most_common(top_n)
        return [(pair[1], count) for pair, count in global_top_locations]

    # Sort the next locations by frequency and return the top N predictions
    return sorted(next_locations.items(), key=lambda x: x[1], reverse=True)[:top_n]


predicted_locations_with_similarity = {}

# Calculate user similarities
user_similarity = calculate_user_similarity(mobility_data)

for uid, group in mobility_data.groupby('uid'):
    last_location = group[['x', 'y']].iloc[-1].values
    last_location = tuple(last_location)
    
    # Predict the top N next locations based on the last location and similar users(in this task 3, N=5)
    top_predictions = predict_top_locations(
        uid, last_location, pair_counts, user_similarity, top_n=5
    )
    
    # Predict the next next location
    next_location = top_predictions[0][0] if top_predictions else None
    next_next_predictions = []
    if next_location:
        next_next_predictions = predict_top_locations(
            uid, next_location, pair_counts, user_similarity, top_n=5
        )
    
    # Predict the next next next location
    next_next_next_predictions = []
    if next_next_predictions:
        next_next_next_predictions = predict_top_locations(
            uid, next_next_predictions[0][0], pair_counts, user_similarity, top_n=5
        )
    
    # Store the predicted locations for the user
    predicted_locations_with_similarity[uid] = {
        'last_known_location': last_location,
        'top_next_locations': top_predictions,
        'top_next_next_locations': next_next_predictions,
        'top_next_next_next_locations': next_next_next_predictions
    }


for uid, locations in predicted_locations_with_similarity.items():
    last_x, last_y = locations['last_known_location']
    print(f"User {uid}:")
    print(f"  Last Known Location: ({int(last_x)}, {int(last_y)})")
    
    if locations['top_next_locations']:
        print("  Top-5 Predicted Next Locations:")
        for rank, (predicted, count) in enumerate(locations['top_next_locations'], start=1):
            predicted_x, predicted_y = predicted
            print(f"    {rank}: ({int(predicted_x)}, {int(predicted_y)}) with frequency {count}")
    
    if locations['top_next_next_locations']:
        print("  Top-5 Predicted Next Next Locations:")
        for rank, (predicted, count) in enumerate(locations['top_next_next_locations'], start=1):
            predicted_x, predicted_y = predicted
            print(f"    {rank}: ({int(predicted_x)}, {int(predicted_y)}) with frequency {count}")
    
    if locations['top_next_next_next_locations']:
        print("  Top-5 Predicted Next Next Next Locations:")
        for rank, (predicted, count) in enumerate(locations['top_next_next_next_locations'], start=1):
            predicted_x, predicted_y = predicted
            print(f"    {rank}: ({int(predicted_x)}, {int(predicted_y)}) with frequency {count}")
    
    print('-' * 50)