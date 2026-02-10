
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


data = {
    'Annual Income (k$)': [15, 16, 17, 18, 19, 20, 21, 60, 62, 63, 64, 65, 66, 67, 120, 122, 125, 127, 130, 135],
    'Spending Score': [39, 81, 6, 77, 40, 76, 6, 40, 42, 43, 40, 42, 43, 44, 15, 16, 17, 12, 15, 20]
}
df = pd.DataFrame(data)


plt.scatter(df['Annual Income (k$)'], df['Spending Score'])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Distribution")
plt.show()


X = df[['Annual Income (k$)', 'Spending Score']]
inertia_values = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()


kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)
print(df.head())


plt.figure(figsize=(8,5))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score'],
    hue=df['Cluster'],
    palette='viridis',
    s=100
)

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='X', label='Centroids')
plt.title("K-Means Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()
plt.show()