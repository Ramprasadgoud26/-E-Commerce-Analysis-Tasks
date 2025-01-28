import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load datasets
transactions_df = pd.read_csv('Transactions.csv')
products_df = pd.read_csv('Products.csv')
customers_df = pd.read_csv('Customers.csv')

# Filter the first 20 customers
first_20_customers = customers_df[customers_df['CustomerID'].isin([f'C{str(i).zfill(4)}' for i in range(1, 21)])]

# Merge with transaction and product data to create a full profile
customer_transactions = transactions_df.merge(products_df, on='ProductID').merge(customers_df, on='CustomerID')

# Create aggregated features for each customer:
# 1. Total transaction value
# 2. Number of transactions
# 3. Most common product category purchased
# 4. Region (as a categorical feature)

customer_features = customer_transactions.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'TransactionID': 'count',
    'Category': lambda x: x.mode()[0],  # Most common category
    'Region': 'first'  # Region stays the same for each customer
}).reset_index()

# Encode categorical features (Region and Category)
customer_features = pd.get_dummies(customer_features, columns=['Category', 'Region'], drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
customer_features[['TotalValue', 'TransactionID']] = scaler.fit_transform(customer_features[['TotalValue', 'TransactionID']])

# Calculate cosine similarity between all customers
customer_vectors = customer_features.drop('CustomerID', axis=1).values
similarity_matrix = cosine_similarity(customer_vectors)

# For each of the first 20 customers, find the top 3 similar customers and their scores
lookalike_map = {}

for idx, customer_id in enumerate(first_20_customers['CustomerID']):
    similarity_scores = similarity_matrix[idx]
    # Get the indices of the top 3 most similar customers (excluding the customer itself)
    top_3_indices = np.argsort(similarity_scores)[::-1][1:4]
    top_3_customers = customer_features.iloc[top_3_indices]['CustomerID'].values
    top_3_scores = similarity_scores[top_3_indices]
    
    # Store the results in a dictionary
    lookalike_map[customer_id] = list(zip(top_3_customers, top_3_scores))

# Convert the lookalike map to a DataFrame for export
lookalike_df = pd.DataFrame([(cust_id, cust_list) for cust_id, cust_list in lookalike_map.items()],
                            columns=['CustomerID', 'Lookalikes'])

lookalike_df['Lookalikes'] = lookalike_df['Lookalikes'].apply(lambda x: [{'cust_id': i[0], 'score': i[1]} for i in x])

# Save to CSV
lookalike_df.to_csv('Lookalike.csv', index=False)
