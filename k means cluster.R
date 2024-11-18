# Load necessary libraries
library(tidyverse)
library(cluster)
library(factoextra)

# Step 1: Load the Dataset
# Ensure the dataset is saved in your working directory
mall_data <- read.csv("C:\\Users\\asus\\OneDrive\\Desktop\\K22ZM\\INT234 Machine learning\\Project\\archive (2)ty\\Mall_Customers.csv")  # Replace with the correct file path

head(mall_data)  # View the first few rows of the dataset
str(mall_data)   # Check the structure

# Step 2: Data Cleaning and Preprocessing
# Remove non-numeric columns like CustomerID and Gender
numeric_data <- mall_data %>% select(-CustomerID, -Gender)

# Check for missing values
sum(is.na(numeric_data))

# Standardize the data
scaled_data <- scale(numeric_data)

# Step 3: Determine Optimal Number of Clusters
# Use the Elbow Method to determine the ideal number of clusters
fviz_nbclust(scaled_data, kmeans, method = "wss") +
  labs(title = "Elbow Method for Optimal Clusters")

# Step 4: Apply K-Means Clustering
# Set the number of clusters (choose based on the Elbow plot)
k <- 3  # Replace with the desired number of clusters
set.seed(123)  # For reproducibility
kmeans_result <- kmeans(scaled_data, centers = k, nstart = 25)

# Add cluster labels to the original dataset
mall_data$Cluster <- as.factor(kmeans_result$cluster)

# Step 5: Visualize Clusters
# PCA for dimensionality reduction and 2D visualization
pca <- prcomp(scaled_data)
pca_data <- as.data.frame(pca$x[, 1:2])  # Use the first two principal components
pca_data$Cluster <- mall_data$Cluster

# Scatter plot of the clusters
ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3, alpha = 0.8) +
  theme_minimal() +
  labs(title = "K-Means Clustering Results (PCA)", x = "Principal Component 1", y = "Principal Component 2")

# Visualize cluster centers
fviz_cluster(kmeans_result, data = scaled_data) +
  labs(title = "Cluster Visualization")

# Step 6: Analyze Clusters
# Calculate the average values for each cluster
cluster_summary <- mall_data %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), mean))
print(cluster_summary)

# Step 7: Save the Results
# Save the dataset with cluster assignments
write.csv(mall_data, "Mall_Customers_with_Clusters.csv", row.names = FALSE)
