import pandas as pd

# Step 1: Data Collection (Assuming you have a dataset with user interactions)
data = pd.read_csv('user_interactions.csv')  # Assuming you have a CSV file with user interactions

# Step 2: Filter data for the user you want to analyze
user_id_to_analyze = 1
user_data = data[data['user_id'] == user_id_to_analyze]

# Step 3: Calculate frequency count of interactions for each category
category_freq = user_data['category'].value_counts()

# Step 4: Calculate weighted sum of time spent on each category (Assuming 'timestamp' column is present)
user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
category_time_spent = user_data.groupby('category')['timestamp'].apply(lambda x: (x.max() - x.min()).total_seconds()).fillna(0)

# Step 5: Combine frequency count and weighted sum
category_preference = category_freq + category_time_spent

# Step 6: Rank categories based on preference
ranked_categories = category_preference.sort_values(ascending=False).index.tolist()

# Step 7: Predicting user's preferred category
# You can choose the most preferred category as the predicted category for the user
predicted_category = ranked_categories[0]

# Print the details of how the program got the predicted category
print("Category preference details:")
for category in ranked_categories:
    freq_count = category_freq.get(category, 0)
    time_spent = category_time_spent.get(category, 0)
    preference_score = category_preference.get(category, 0)
    print(f"Category: {category}, Frequency Count: {freq_count}, Time Spent: {time_spent} seconds, Preference Score: {preference_score}")

print("\nPredicted category for user", user_id_to_analyze, ":", predicted_category)
