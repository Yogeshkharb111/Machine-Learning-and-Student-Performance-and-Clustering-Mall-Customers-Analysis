# Mall Customer Behavior Analysis: Machine Learning & Clustering

## Overview

This project is an analysis of mall customer data using machine learning and clustering techniques. It involves applying **Supervised Learning** (Linear Regression, Support Vector Machine, and Naive Bayes) to predict customer behavior and **Unsupervised Learning** (K-means clustering) to segment customers based on their attributes. 

The project integrates a **Shiny dashboard** to allow users to interact with the model performance results and clustering output, enabling a deeper understanding of how customer behavior can be modeled and segmented for better business decisions.

## Motivation

In the retail industry, understanding customer behavior is crucial for targeting the right audience, improving marketing strategies, and optimizing the product offering. This project helps analyze various aspects of customer behavior by predicting their spending patterns and clustering them based on common characteristics.

By applying machine learning and clustering techniques, we aim to:
- Predict customer spending based on their demographic and behavioral characteristics.
- Segment customers into meaningful groups for targeted marketing campaigns.
- Provide insights into the effectiveness of various machine learning models for these tasks.

## Methodology

The project applies the following machine learning models and techniques:

### 1. **Supervised Learning**
- **Linear Regression**: Used to predict the spending score (or a continuous variable) based on other numeric features like age and annual income.
- **Support Vector Machine (SVM)**: A classification technique that categorizes customers based on their spending behavior.
- **Naive Bayes**: A probabilistic classifier to categorize customers into performance levels (Low, Medium, High) based on their spending scores.

### 2. **Unsupervised Learning**
- **K-means Clustering**: A clustering technique used to segment customers into different groups based on their demographic information and spending behavior. The number of clusters is determined using the **Elbow Method**.

### 3. **Dimensionality Reduction**
- **Principal Component Analysis (PCA)**: Applied to reduce the dimensionality of the dataset for easier visualization and interpretation of clusters.

### 4. **Shiny Dashboard**
- The app allows users to:
  - Compare the performance of the three models.
  - Visualize the **True vs Predicted** plots for regression models.
  - Explore **Performance Distribution** for Naive Bayes.
  - View **Cluster Visualization** and segment customers based on clusters.

## Dataset

The project uses the **Mall Customer Dataset**, which includes the following features:
- **CustomerID**: Unique identifier for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Annual Income**: Annual income of the customer.
- **Spending Score**: A score given to customers based on their spending behavior.

### Dataset Preprocessing:
- Non-numeric columns (such as `CustomerID` and `Gender`) are removed or encoded as needed.
- Missing values are checked and appropriately handled.
- Data is scaled to standardize the features for clustering and regression models.

## Key Features

- **Predictive Modeling**:
  - Compare the performance of **Linear Regression**, **SVM**, and **Naive Bayes**.
  - Predict spending behavior and categorize customers based on their scores.
  
- **Customer Segmentation**:
  - Apply **K-means clustering** to segment customers into different groups.
  - Visualize clusters using **PCA** for dimensionality reduction.

- **Shiny Dashboard**:
  - Interactive visualizations of model metrics, predictions, and performance.
  - Real-time comparison of model predictions and actual values for better insights.

## Requirements

Before running the project, ensure the following R packages are installed:

- `shiny`: For creating the interactive dashboard.
- `ggplot2`: For visualizing the data and model outputs.
- `caret`: For training machine learning models.
- `e1071`: For SVM and Naive Bayes implementations.
- `dplyr`: For data manipulation.
- `tidyverse`: For general data manipulation and visualization.
- `cluster`: For clustering techniques.
- `factoextra`: For visualizing clustering results and determining optimal cluster numbers.

To install the required packages, run:

```r
install.packages(c("shiny", "ggplot2", "caret", "e1071", "dplyr", "tidyverse", "cluster", "factoextra"))
