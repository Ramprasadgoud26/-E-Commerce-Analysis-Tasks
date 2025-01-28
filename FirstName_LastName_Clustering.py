import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
transactions_df = pd.read_csv('Transactions.csv')
products_df = pd.read_csv('Products.csv')
customers_df = pd.read_csv('Customers.csv')

# Prepare data by merging customer and transaction data
customer_transactions = transactions_df.merge(products_df, on='ProductID').merge(customers_df, on='CustomerID')

# Create aggregated features for each customer:
# 1. Total transaction value
# 2. Number of transactions
# 3. Region as a categorical feature
customer_features_clustering = customer_transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'TransactionID': 'count',
    'Region': 'first'  # Region stays the same for each customer
}).reset_index()

# Encode categorical features (Region)
customer_features_clustering = pd.get_dummies(customer_features_clustering, columns=['Region'], drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
customer_features_clustering[['TotalValue', 'TransactionID']] = scaler.fit_transform(customer_features_clustering[['TotalValue', 'TransactionID']])

# Apply KMeans clustering for a range of clusters (2 to 10) and calculate Davies-Bouldin Index
db_indices = []
cluster_range = range(2, 11)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(customer_features_clustering.drop('CustomerID', axis=1))
    
    # Calculate DB Index for the clustering
    db_index = davies_bouldin_score(customer_features_clustering.drop('CustomerID', axis=1), cluster_labels)
    db_indices.append(db_index)

# Find the best number of clusters (lowest DB index)
best_n_clusters = cluster_range[np.argmin(db_indices)]
best_db_index = min(db_indices)

# Refit KMeans with the best number of clusters
kmeans_best = KMeans(n_clusters=best_n_clusters, random_state=42)
customer_features_clustering['Cluster'] = kmeans_best.fit_predict(customer_features_clustering.drop('CustomerID', axis=1))

# Visualization of clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_features_clustering, x='TotalValue', y='TransactionID', hue='Cluster', palette='viridis', s=100)
plt.title(f'Customer Clusters (Best: {best_n_clusters} Clusters, DB Index: {best_db_index:.2f})')
plt.xlabel('Total Transaction Value (Standardized)')
plt.ylabel('Number of Transactions (Standardized)')
plt.show()
