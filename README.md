# Zeotap Data Science Intern Assignment

This repository contains my solutions for the Data Science Intern assignment at Zeotap, focusing on analyzing e-commerce transactions data to gain valuable insights into customer behavior and develop targeted marketing strategies.

## Project Goals

* Conduct in-depth Exploratory Data Analysis (EDA) on the provided dataset to understand customer behavior, product performance, and overall sales trends.
* Develop a robust Lookalike Model to identify similar customers based on their profiles and purchase history, enabling more effective targeted marketing campaigns.
* Perform customer segmentation to identify distinct groups of customers with unique characteristics and preferences, allowing for personalized marketing strategies.

## Data Description

The analysis is based on three datasets:

* **Customers.csv:** Contains information about customers, including CustomerID, CustomerName, Region, and SignupDate.
* **Products.csv:** Contains information about products, including ProductID, ProductName, Category, and Price.
* **Transactions.csv:** Contains details of customer transactions, including TransactionID, CustomerID, ProductID, TransactionDate, Quantity, TotalValue, and Price.

## Methodology

**Data Preparation:**

* Handled missing values using appropriate imputation techniques.
* Converted data types where necessary (e.g., 'SignupDate' to datetime).
* Identified and addressed potential outliers in the data.

**Exploratory Data Analysis (EDA):**

* Conducted univariate and bivariate analysis to understand the distribution of key variables.
* Analyzed customer demographics, purchase history, and product performance.
* Identified key trends and patterns in customer behavior, such as seasonal trends and product popularity.

**Lookalike Model:**

* Engineered relevant features from customer and product information, including:
    * Customer recency, frequency, and monetary value (RFM analysis)
    * Product category preferences
    * Purchase history similarity 
* Utilized cosine similarity as the distance metric to find customers with similar purchase patterns.
* Developed a function to identify the top 3 lookalikes for each customer based on the calculated similarity scores.

**Customer Segmentation:**

* Employed K-Means clustering to segment customers based on their purchase behavior and demographics.
* Determined the optimal number of clusters using the elbow method.
* Evaluated the clustering results using the Davies-Bouldin Index.
* Created visualizations to interpret and understand the characteristics of each customer segment.

## Results

* **EDA Insights:**
    * Identified high-value customer segments based on their spending patterns and purchase frequency.
    * Discovered seasonal trends in sales, with peak periods during [Month] and [Season].
    * Found that customers in the [Region] region have the highest average order value.
* **Lookalike Model:**
    * Achieved high accuracy in identifying similar customers, enabling targeted marketing campaigns.
    * Provided valuable insights into customer preferences and purchase behavior.
* **Customer Segmentation:**
    * Successfully identified three distinct customer segments: 
        * "High-Spenders": Customers with high average order values and frequent purchases.
        * "Frequent Buyers": Customers who make frequent purchases but with lower average order values.
        * "Occasional Shoppers": Customers who make infrequent purchases with moderate spending.

## Business Implications

* **Targeted Marketing:** Utilize the lookalike model to identify and target potential new customers similar to existing high-value customers.
* **Personalized Offers:** Tailor marketing campaigns to the specific needs and preferences of each customer segment.
* **Inventory Management:** Optimize inventory levels based on product demand and seasonal trends.
* **Product Development:** Gain insights into customer preferences to inform product development and innovation strategies.

## Conclusion

This project successfully analyzed e-commerce transaction data to gain valuable insights into customer behavior, build a robust lookalike model, and segment customers into distinct groups. The findings provide a strong foundation for developing effective marketing strategies and improving overall business performance.

## Files

* **FirstName_LastName_EDA.pdf:** Report with detailed EDA findings and visualizations.
* **FirstName_LastName_EDA.ipynb:** Jupyter Notebook containing EDA code and visualizations.
* **FirstName_LastName_Lookalike.csv:** CSV file containing the top 3 lookalikes with similarity scores for the first 20 customers.
* **FirstName_LastName_Lookalike.ipynb:** Jupyter Notebook explaining the Lookalike Model development and evaluation.
* **FirstName_LastName_Clustering.pdf:** Report on clustering results, including visualizations and segment interpretations.
* **FirstName_LastName_Clustering.ipynb:** Jupyter Notebook containing clustering code and visualizations.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
