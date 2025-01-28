import pandas as pd

# Load datasets
transactions_df = pd.read_csv('Transactions.csv')
products_df = pd.read_csv('Products.csv')
customers_df = pd.read_csv('Customers.csv')

# Merge datasets for combined analysis
transactions_merged = transactions_df.merge(products_df, on='ProductID').merge(customers_df, on='CustomerID')

# Top-selling products
top_selling_products = transactions_merged.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(5)

# Customer region distribution
customer_region_distribution = customers_df['Region'].value_counts()

# Average transaction value by region
avg_transaction_value_by_region = transactions_merged.groupby('Region')['TotalValue'].mean()

# Popular product categories
popular_categories = transactions_merged.groupby('Category')['Quantity'].sum().sort_values(ascending=False).head(5)

# Sales trends over time
transactions_merged['TransactionDate'] = pd.to_datetime(transactions_merged['TransactionDate'])
transactions_merged['Month'] = transactions_merged['TransactionDate'].dt.to_period('M')
sales_trends_over_time = transactions_merged.groupby('Month')['TotalValue'].sum()

# Output EDA results
print('Top Selling Products:')
print(top_selling_products)
print('\\nCustomer Region Distribution:')
print(customer_region_distribution)
print('\\nAverage Transaction Value by Region:')
print(avg_transaction_value_by_region)
print('\\nPopular Product Categories:')
print(popular_categories)
print('\\nSales Trends Over Time:')
print(sales_trends_over_time)
