import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load prepared data
prepared_data_path = 'prepared_data.csv'
data_cm = pd.read_csv(prepared_data_path)

# Selecting features for clustering
features = data_cm[['Jenis Gangguan', 'Penyebab', 'ZONA', 'Merk Modem']]

# Elbow Method to find optimal K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

# Plot WCSS to find the "elbow"
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Perform K-Means clustering with optimal K (e.g., K=3)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(features)

# Evaluate the model
sil_score = silhouette_score(features, y_kmeans)
inertia = kmeans.inertia_
db_index = davies_bouldin_score(features, y_kmeans)

# Display evaluation metrics
print(f"Silhouette Score: {sil_score}")
print(f"Inertia (WCSS): {inertia}")
print(f"Davies-Bouldin Index: {db_index}")

# Save the clustering results to the dataset
data_cm['Cluster'] = y_kmeans

# Save clustered data to CSV
data_cm.to_csv('clustered_data.csv', index=False)
print("Clustered data saved to 'clustered_data.csv'")
