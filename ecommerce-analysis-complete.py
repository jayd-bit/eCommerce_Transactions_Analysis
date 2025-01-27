# main.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class EcommerceAnalyzer:
    def __init__(self, data_dir='data', output_dir='output'):
        """Initialize the analyzer with data and output directories."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.customers = None
        self.products = None
        self.transactions = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_data(self):
        """Load data from CSV files."""
        try:
            self.customers = pd.read_csv(f'{self.data_dir}/Customers.csv')
            self.products = pd.read_csv(f'{self.data_dir}/Products.csv')
            self.transactions = pd.read_csv(f'{self.data_dir}/Transactions.csv')
            
            # Convert date columns
            self.customers['SignupDate'] = pd.to_datetime(self.customers['SignupDate'])
            self.transactions['TransactionDate'] = pd.to_datetime(self.transactions['TransactionDate'])
            
            print("Data loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_rfm(self, reference_date=None):
        """Calculate RFM metrics for each customer."""
        if reference_date is None:
            reference_date = self.transactions['TransactionDate'].max()
        
        rfm = self.transactions.groupby('CustomerID').agg({
            'TransactionDate': lambda x: (reference_date - x.max()).days,  # Recency
            'TransactionID': 'count',  # Frequency
            'TotalValue': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        return rfm
    
    def create_customer_features(self):
        """Create feature matrix for customer analysis."""
        # Calculate RFM
        rfm = self.calculate_rfm()
        
        # Calculate category preferences
        merged_data = self.transactions.merge(self.products, on='ProductID')
        category_preferences = merged_data.groupby(['CustomerID', 'Category'])['Quantity'].sum().unstack(fill_value=0)
        
        # Combine features
        features = rfm.merge(category_preferences, on='CustomerID')
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features.drop('CustomerID', axis=1))
        
        return features['CustomerID'], scaled_features
    
    def find_lookalikes(self, n_recommendations=3):
        """Find similar customers using cosine similarity."""
        customer_ids, features = self.create_customer_features()
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(features)
        
        # Find top N similar customers
        lookalikes = []
        for i, customer_id in enumerate(customer_ids):
            similarities = similarity_matrix[i]
            similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
            similar_customers = customer_ids.iloc[similar_indices]
            similar_scores = similarities[similar_indices]
            
            for similar_id, score in zip(similar_customers, similar_scores):
                lookalikes.append({
                    'CustomerID': customer_id,
                    'SimilarCustomerID': similar_id,
                    'SimilarityScore': score
                })
        
        return pd.DataFrame(lookalikes)
    
    def segment_customers(self, n_clusters=3):
        """Perform customer segmentation using K-means clustering."""
        customer_ids, features = self.create_customer_features()
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Calculate cluster quality
        quality_score = davies_bouldin_score(features, clusters)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'CustomerID': customer_ids,
            'Cluster': clusters
        })
        
        return results, quality_score
    
    def analyze_sales(self):
        """Analyze sales patterns and trends."""
        # Merge data
        sales_data = self.transactions.merge(self.customers, on='CustomerID')
        sales_data = sales_data.merge(self.products, on='ProductID')
        
        # Calculate key metrics
        metrics = {
            'total_sales': sales_data['TotalValue'].sum(),
            'total_customers': len(sales_data['CustomerID'].unique()),
            'avg_order_value': sales_data['TotalValue'].mean(),
            'total_products': len(sales_data['ProductID'].unique()),
            'top_categories': sales_data.groupby('Category')['TotalValue'].sum().sort_values(ascending=False),
            'sales_by_month': sales_data.groupby(sales_data['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()
        }
        
        return metrics
    
    def generate_visualizations(self):
        """Generate and save visualization plots."""
        # Sales trends
        plt.figure(figsize=(15, 10))
        
        # Monthly sales
        plt.subplot(2, 2, 1)
        monthly_sales = self.transactions.groupby(
            self.transactions['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()
        monthly_sales.plot()
        plt.title('Monthly Sales Trend')
        
        # Category distribution
        plt.subplot(2, 2, 2)
        merged_data = self.transactions.merge(self.products, on='ProductID')
        category_sales = merged_data.groupby('Category')['TotalValue'].sum()
        category_sales.plot(kind='pie')
        plt.title('Sales by Category')
        
        # Customer regions
        plt.subplot(2, 2, 3)
        self.customers['Region'].value_counts().plot(kind='bar')
        plt.title('Customers by Region')
        
        # Order value distribution
        plt.subplot(2, 2, 4)
        plt.hist(self.transactions['TotalValue'], bins=50)
        plt.title('Order Value Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/analysis_visualizations.png')
        plt.close()
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        # Load data
        if not self.load_data():
            return False
        
        try:
            # Generate lookalikes
            lookalikes = self.find_lookalikes()
            lookalikes.to_csv(f'{self.output_dir}/lookalikes.csv', index=False)
            
            # Perform segmentation
            segments, quality_score = self.segment_customers()
            segments.to_csv(f'{self.output_dir}/customer_segments.csv', index=False)
            
            # Analyze sales
            metrics = self.analyze_sales()
            
            # Save metrics to file
            with open(f'{self.output_dir}/analysis_metrics.txt', 'w') as f:
                for metric, value in metrics.items():
                    f.write(f"{metric}:\n{value}\n\n")
            
            # Generate visualizations
            self.generate_visualizations()
            
            print("Analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return False

def main():
    # Initialize analyzer
    analyzer = EcommerceAnalyzer(data_dir='data', output_dir='output')
    
    # Run analysis
    success = analyzer.run_analysis()
    
    if success:
        print("Analysis completed successfully. Check the output directory for results.")
    else:
        print("Analysis failed. Check the error messages above.")

if __name__ == "__main__":
    main()
