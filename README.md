# Machine Learning and Clustering: Mall Customers Analysis

This project combines machine learning and clustering techniques to analyze and predict customer behavior based on data from a mall customer dataset. The project applies **Linear Regression**, **SVM**, and **Naive Bayes** models for predictive analysis, and uses **K-means clustering** to segment customers based on purchasing behavior. The results are showcased in an interactive **Shiny dashboard** that allows users to compare model performance and explore clustering results.

## Project Overview

The main objectives of this project are:
- To predict customer behavior using different machine learning models (Linear Regression, SVM, and Naive Bayes).
- To segment customers into distinct clusters using K-means clustering.
- To provide an interactive dashboard for visualizing the performance of the models and clustering results.

## Datasets

The dataset used in this project is a CSV file of mall customer information. It includes:
- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

## Key Features
- **Data Preprocessing**: Cleaning and transforming the dataset to prepare it for modeling.
- **Modeling**: 
  - Linear Regression, SVM, and Naive Bayes for predictive analytics.
  - K-means clustering for unsupervised customer segmentation.
- **Visualization**: Various plots to visualize the data distribution, correlations, and clustering results.
- **Shiny App**: A dashboard for comparing model performance and analyzing clustering results interactively.

## Requirements

To run this project, you'll need to install the following R packages:
- `shiny`
- `ggplot2`
- `caret`
- `e1071`
- `dplyr`
- `tidyverse`
- `cluster`
- `factoextra`

You can install them using the following commands:

```r
install.packages(c("shiny", "ggplot2", "caret", "e1071", "dplyr", "tidyverse", "cluster", "factoextra"))
