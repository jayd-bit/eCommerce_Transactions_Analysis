# ecommerce_analysis.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import joblib
import os
from typing import Tuple, Dict, List, Any, Optional

class DataValidator:
    """Validates and cleans input data."""
    
    @staticmethod
    def validate_dates(df: pd.DataFrame, date_columns: List[str]) -> bool:
        """Validate date columns."""
        try:
            for col in date_columns:
                df[col] = pd.to_datetime(df[col])
            return True
        except Exception as e:
            logging.error(f"Date validation error: {e}")
            return False
    
    @staticmethod
    def validate_numeric(df: pd.DataFrame, numeric_columns: List[str]) -> bool:
        """Validate numeric columns."""
        try:
            for col in numeric_columns:
                pd.to_numeric(df[col])
            return True
        except Exception as e:
            logging.error(f"Numeric validation error: {e}")
            return False
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method."""
        df_clean = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df_clean = df_clean[
                    (df_clean[col] >= Q1 - 1.5 * IQR) &
                    (df_clean[col] <= Q3 + 1.5 * IQR)
                ]
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df_clean = df_clean[z_scores < 3]
                
        return df_clean

class FeatureEngineering:
    """Handles feature engineering and preprocessing."""
    
    @staticmethod
    def calculate_rfm(transactions: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """Calculate RFM metrics with proper scaling."""
        if reference_date is None:
            reference_date = transactions['TransactionDate'].max()
            
        rfm = transactions.groupby('CustomerID').agg({
            'TransactionDate': lambda x: (reference_date - x.max()).days,  # Recency
            'TransactionID': 'count',  # Frequency
            'TotalValue': ['sum', 'mean', 'std']  # Monetary metrics
        }).reset_index()
        
        # Flatten column names
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'TotalValue', 'AvgValue', 'StdValue']
        
        # Log transform skewed features
        for col in ['Frequency', 'TotalValue', 'AvgValue']:
            rfm[f'{col}_log'] = np.log1p(rfm[col])
            
        return rfm
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create time-based features."""
        df = df.copy()
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        return df
    
    @staticmethod
    def calculate_customer_lifetime_value(transactions: pd.DataFrame) -> pd.Series:
        """Calculate customer lifetime value."""
        customer_history = transactions.groupby('CustomerID').agg({
            'TransactionDate': lambda x: (x.max() - x.min()).days,
            'TotalValue': 'sum'
        })
        
        # Calculate daily rate and project annual value
        customer_history['CLV'] = (customer_history['TotalValue'] / 
                                 np.maximum(customer_history['TransactionDate'], 1)) * 365
        return customer_history['CLV']

class CustomerSegmentation:
    """Handles customer segmentation with multiple algorithms."""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)  # Preserve 95% variance
        self.best_model = None
        
    def prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare features for clustering."""
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Apply PCA
        reduced_features = self.pca.fit_transform(scaled_features)
        return reduced_features
    
    def find_optimal_clusters(self, features: np.ndarray, max_clusters: int = 10) -> Dict[str, List[float]]:
        """Find optimal number of clusters using multiple metrics."""
        metrics = {
            'silhouette': [],
            'calinski': [],
            'davies': []
        }
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            metrics['silhouette'].append(silhouette_score(features, labels))
            metrics['calinski'].append(calinski_harabasz_score(features, labels))
            metrics['davies'].append(davies_bouldin_score(features, labels))
            
        return metrics
    
    def segment_customers(self, features: pd.DataFrame, n_clusters: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """Perform customer segmentation."""
        prepared_features = self.prepare_features(features)
        
        if n_clusters is None:
            # Find optimal clusters
            metrics = self.find_optimal_clusters(prepared_features)
            n_clusters = np.argmax(metrics['silhouette']) + 2
        
        # Try both KMeans and DBSCAN
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        # Compare results
        kmeans_labels = kmeans.fit_predict(prepared_features)
        dbscan_labels = dbscan.fit_predict(prepared_features)
        
        kmeans_score = silhouette_score(prepared_features, kmeans_labels)
        try:
            dbscan_score = silhouette_score(prepared_features, dbscan_labels)
        except:
            dbscan_score = -1
        
        # Use the better performing model
        if kmeans_score > dbscan_score:
            self.best_model = kmeans
            return kmeans_labels, kmeans_score
        else:
            self.best_model = dbscan
            return dbscan_labels, dbscan_score

class LookalikeModel:
    """Handles customer lookalike analysis."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_similarity(self, features: pd.DataFrame) -> np.ndarray:
        """Calculate customer similarity with proper scaling."""
        scaled_features = self.scaler.fit_transform(features)
        return cosine_similarity(scaled_features)
    
    def find_lookalikes(self, features: pd.DataFrame, customer_ids: pd.Series, 
                       n_recommendations: int = 3, min_similarity: float = 0.5) -> pd.DataFrame:
        """Find lookalike customers with validation."""
        similarity_matrix = self.calculate_similarity(features)
        
        lookalikes = []
        for i, customer_id in enumerate(customer_ids):
            similarities = similarity_matrix[i]
            # Get top N+1 and remove self
            similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
            similar_customers = customer_ids.iloc[similar_indices]
            similar_scores = similarities[similar_indices]
            
            # Only include recommendations above minimum similarity
            for similar_id, score in zip(similar_customers, similar_scores):
                if score >= min_similarity:
                    lookalikes.append({
                        'CustomerID': customer_id,
                        'SimilarCustomerID': similar_id,
                        'SimilarityScore': score
                    })
        
        return pd.DataFrame(lookalikes)

class EcommerceAnalyzer:
    """Main analysis class."""
    
    def __init__(self, data_dir: str = 'data', output_dir: str = 'output'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.validator = DataValidator()
        self.feature_engineering = FeatureEngineering()
        self.segmentation = CustomerSegmentation()
        self.lookalike = LookalikeModel()
        
        # Set up logging
        logging.basicConfig(
            filename=f'{output_dir}/analysis.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and validate all datasets."""
        try:
            # Load data
            customers = pd.read_csv(f'{self.data_dir}/Customers.csv')
            products = pd.read_csv(f'{self.data_dir}/Products.csv')
            transactions = pd.read_csv(f'{self.data_dir}/Transactions.csv')
            
            # Validate dates
            if not self.validator.validate_dates(customers, ['SignupDate']):
                raise ValueError("Invalid customer dates")
            if not self.validator.validate_dates(transactions, ['TransactionDate']):
                raise ValueError("Invalid transaction dates")
            
            # Validate numeric columns
            numeric_cols = ['TotalValue', 'Quantity', 'Price']
            if not self.validator.validate_numeric(transactions, numeric_cols):
                raise ValueError("Invalid numeric values")
            
            # Remove outliers
            transactions = self.validator.remove_outliers(
                transactions, ['TotalValue', 'Quantity'], method='iqr'
            )
            
            return customers, products, transactions
            
        except Exception as e:
            logging.error(f"Data loading error: {e}")
            raise
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline with validation."""
        try:
            # Load and validate data
            customers, products, transactions = self.load_and_validate_data()
            logging.info("Data loaded and validated successfully")
            
            # Feature engineering
            rfm = self.feature_engineering.calculate_rfm(transactions)
            time_features = self.feature_engineering.create_time_features(transactions, 'TransactionDate')
            clv = self.feature_engineering.calculate_customer_lifetime_value(transactions)
            
            # Create feature matrix
            features = rfm.merge(pd.DataFrame(clv), on='CustomerID')
            logging.info("Features engineered successfully")
            
            # Perform segmentation
            labels, score = self.segmentation.segment_customers(
                features.drop('CustomerID', axis=1)
            )
            segments = pd.DataFrame({
                'CustomerID': features['CustomerID'],
                'Segment': labels
            })
            logging.info(f"Segmentation completed with score: {score}")
            
            # Find lookalikes
            lookalikes = self.lookalike.find_lookalikes(
                features.drop('CustomerID', axis=1),
                features['CustomerID']
            )
            logging.info("Lookalike analysis completed")
            
            # Save results
            results = {
                'segments': segments,
                'lookalikes': lookalikes,
                'rfm': rfm,
                'clv': clv,
                'segmentation_score': score
            }
            
            self.save_results(results)
            logging.info("Analysis completed successfully")
            
            return results
            
        except Exception as e:
            logging.error(f"Analysis error: {e}")
            raise
    
    def save_results(self, results: Dict[str, Any]):
        """Save analysis results."""
        try:
            for name, data in results.items():
                if isinstance(data, pd.DataFrame):
                    data.to_csv(f'{self.output_dir}/{name}.csv', index=False)
                elif isinstance(data, pd.Series):
                    data.to_csv(f'{self.output_dir}/{name}.csv')
                else:
                    with open(f'{self.output_dir}/{name}.txt', 'w') as f:
                        f.write(str(data))
            
            # Save models
            joblib.dump(self.segmentation.best_model, f'{self.output_dir}/segmentation_model.pkl')
            
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            raise

def main():
    analyzer = EcommerceAnalyzer()
    try:
        results = analyzer.run_analysis()
        print("Analysis completed successfully. Check the output directory for results.")
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()
