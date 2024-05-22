import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load event data from CSV file
events_data = []
with open('event.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        events_data.append(row)

user_preferences = {"category": "sports"}

event_categories = [event['category'] for event in events_data]
tfidf_vectorizer = TfidfVectorizer()
event_category_vectors = tfidf_vectorizer.fit_transform(event_categories)

user_category_vector = tfidf_vectorizer.transform([user_preferences['category']])

knn_classifier = KNeighborsClassifier(n_neighbors=len(events_data), metric='cosine')
knn_classifier.fit(event_category_vectors, range(len(events_data)))

nearest_neighbors_indices = knn_classifier.kneighbors(user_category_vector, return_distance=False)
nearest_neighbors_distances, _ = knn_classifier.kneighbors(user_category_vector)

cosine_similarities = cosine_similarity(user_category_vector, event_category_vectors)

# Normalize cosine similarities
cosine_similarities_normalized = cosine_similarities / np.linalg.norm(cosine_similarities)

# Normalize KNN distances
knn_distances_normalized = 1 / (1 + nearest_neighbors_distances)

# Calculate weighted similarity scores
weighted_similarity_scores = (cosine_similarities_normalized + knn_distances_normalized) / 2

# Get indices of events sorted by cosine similarity (descending order)
sorted_indices = cosine_similarities.argsort()[0][::-1]

# Recommend events based on highest cosine similarity
recommended_events = [events_data[i] for i in sorted_indices]

print("Recommended events based on user preferences (category) and cosine similarity:")
for event, cosine_similarity_value in zip(recommended_events, cosine_similarities[0][sorted_indices]):
    print(f"{event['title']} - {event['description']} - Cosine Similarity: {cosine_similarity_value}")

print("\nCosine Similarities:", cosine_similarities)
print("KNN Distances:", nearest_neighbors_distances)
print("Normalized Cosine Similarities:", cosine_similarities_normalized)
print("Normalized KNN Distances:", knn_distances_normalized)
print("Weighted Similarity Scores:", weighted_similarity_scores)