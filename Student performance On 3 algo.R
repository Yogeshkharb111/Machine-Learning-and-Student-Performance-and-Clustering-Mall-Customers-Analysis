# Install required packages if not already installed
if (!require(shiny)) install.packages("shiny")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(caret)) install.packages("caret")
if (!require(e1071)) install.packages("e1071")
if (!require(dplyr)) install.packages("dplyr")

# Load required libraries
library(shiny)
library(ggplot2)
library(caret)
library(e1071)  # For SVM and Naive Bayes
library(dplyr)

# Load the Dataset (Ensure the file path is correct)
dataset <- read.csv("C:\\Users\\asus\\OneDrive\\Desktop\\K22ZM\\INT234 Machine learning\\Project\\archive\\StudentsPerformance.csv")

# Preprocess the Data
colnames(dataset) <- c("gender", "race", "parental_education", 
                       "lunch", "test_preparation", 
                       "math_score", "reading_score", "writing_score")

# Convert categorical variables to factors
dataset$gender <- as.factor(dataset$gender)
dataset$race <- as.factor(dataset$race)
dataset$parental_education <- as.factor(dataset$parental_education)
dataset$lunch <- as.factor(dataset$lunch)
dataset$test_preparation <- as.factor(dataset$test_preparation)

# Check for missing values
sum(is.na(dataset))  # Should return 0 if there are no missing values

# Basic EDA
summary(dataset)  # Summary statistics for all variables
str(dataset)  # Structure of the data (e.g., data types)

# Visualize the distribution of numeric variables (Math, Reading, Writing Scores)
ggplot(dataset, aes(x = math_score)) + 
  geom_histogram(binwidth = 5, fill = "blue", color = "black") + 
  labs(title = "Math Score Distribution", x = "Math Score", y = "Frequency")

ggplot(dataset, aes(x = reading_score)) + 
  geom_histogram(binwidth = 5, fill = "green", color = "black") + 
  labs(title = "Reading Score Distribution", x = "Reading Score", y = "Frequency")

ggplot(dataset, aes(x = writing_score)) + 
  geom_histogram(binwidth = 5, fill = "red", color = "black") + 
  labs(title = "Writing Score Distribution", x = "Writing Score", y = "Frequency")

# Visualize correlations between scores
ggplot(dataset, aes(x = reading_score, y = writing_score)) +
  geom_point(color = "blue") + 
  labs(title = "Reading vs Writing Score", x = "Reading Score", y = "Writing Score") +
  geom_smooth(method = "lm", color = "red")

ggplot(dataset, aes(x = math_score, y = reading_score)) +
  geom_point(color = "green") + 
  labs(title = "Math vs Reading Score", x = "Math Score", y = "Reading Score") +
  geom_smooth(method = "lm", color = "red")

ggplot(dataset, aes(x = math_score, y = writing_score)) +
  geom_point(color = "red") + 
  labs(title = "Math vs Writing Score", x = "Math Score", y = "Writing Score") +
  geom_smooth(method = "lm", color = "red")

# Visualize categorical variables
ggplot(dataset, aes(x = gender)) +
  geom_bar(fill = "purple") +
  labs(title = "Gender Distribution", x = "Gender", y = "Count")

ggplot(dataset, aes(x = race)) +
  geom_bar(fill = "orange") +
  labs(title = "Race Distribution", x = "Race", y = "Count")

ggplot(dataset, aes(x = parental_education)) +
  geom_bar(fill = "brown") +
  labs(title = "Parental Education Distribution", x = "Parental Education", y = "Count")

ggplot(dataset, aes(x = lunch)) +
  geom_bar(fill = "yellow") +
  labs(title = "Lunch Distribution", x = "Lunch", y = "Count")

ggplot(dataset, aes(x = test_preparation)) +
  geom_bar(fill = "pink") +
  labs(title = "Test Preparation Distribution", x = "Test Preparation", y = "Count")

# Correlation Matrix for numeric variables (Math, Reading, Writing Scores)
cor_matrix <- cor(dataset[, c("math_score", "reading_score", "writing_score")])
print(cor_matrix)


# Split the Data into Training and Test sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(dataset$math_score, p = 0.8, list = FALSE)
train_data <- dataset[trainIndex, ]
test_data <- dataset[-trainIndex, ]

# Linear Regression Model
lm_model <- lm(math_score ~ reading_score + writing_score, data = train_data)
lm_predictions <- predict(lm_model, test_data)
lm_mse <- mean((lm_predictions - test_data$math_score)^2)  # Mean Squared Error for Linear Regression

# Support Vector Machine Model
svm_model <- svm(math_score ~ reading_score + writing_score, data = train_data, kernel = "linear")
svm_predictions <- predict(svm_model, test_data)
svm_mse <- mean((svm_predictions - test_data$math_score)^2)  # Mean Squared Error for SVM

# Naive Bayes Model (Classifying Performance)
train_data <- train_data %>%
  mutate(performance = cut(math_score, 
                           breaks = c(-Inf, 60, 80, Inf), 
                           labels = c("Low", "Medium", "High")))
test_data <- test_data %>%
  mutate(performance = cut(math_score, 
                           breaks = c(-Inf, 60, 80, Inf), 
                           labels = c("Low", "Medium", "High")))

# Train Naive Bayes model
nb_model <- naiveBayes(performance ~ gender + reading_score + writing_score, data = train_data)

# Predict on test data
nb_predictions <- predict(nb_model, test_data)

# Confusion Matrix for Naive Bayes
nb_cm <- confusionMatrix(nb_predictions, test_data$performance)

# Results
cat("Linear Regression MSE:", lm_mse, "\n")
cat("SVM MSE:", svm_mse, "\n")
cat("Naive Bayes Accuracy:", nb_cm$overall['Accuracy'] * 100, "%\n")



# Shiny UI
library(shiny)
ui <- fluidPage(
  titlePanel("Predictive Analysis Dashboard: Students Performance"),
  sidebarLayout(
    sidebarPanel(
      h4("Model Performance"),
      p("Compare the performance of three predictive models: Linear Regression, SVM, and Naive Bayes."),
      selectInput("model_type", "Choose a Model:", 
                  choices = c("Linear Regression", "SVM", "Naive Bayes"), 
                  selected = "Linear Regression")
    ),
    mainPanel(
      h3("Model Metrics"),
      textOutput("model_metrics"),
      plotOutput("true_vs_predicted"),
      plotOutput("performance_distribution")
    )
  )
)

# Shiny Server
server <- function(input, output) {
  
  # Display Model Metrics
  output$model_metrics <- renderText({
    if (input$model_type == "Linear Regression") {
      paste("Linear Regression MSE:", round(lm_mse, 2))
    } else if (input$model_type == "SVM") {
      paste("SVM MSE:", round(svm_mse, 2))
    } else {paste("Naive Bayes Accuracy:", round(nb_cm$overall['Accuracy'] * 100, 2), "%")
    }
  })
  
  # Plot True vs Predicted for Regression Models
  output$true_vs_predicted <- renderPlot({
    if (input$model_type == "Linear Regression") {
      ggplot() +
        geom_point(aes(x = test_data$math_score, y = lm_predictions), color = "blue") +
        labs(title = "Linear Regression: True vs Predicted", 
             x = "True Math Score", y = "Predicted Math Score") +
        geom_abline(color = "red", linetype = "dashed")
    } else if (input$model_type == "SVM") {
      ggplot() +
        geom_point(aes(x = test_data$math_score, y = svm_predictions), color = "green") +
        labs(title = "SVM: True vs Predicted", 
             x = "True Math Score", y = "Predicted Math Score") +
        geom_abline(color = "red", linetype = "dashed")
    }
  }) # Plot Performance Distribution for Naive Bayes
  output$performance_distribution <- renderPlot({
    if (input$model_type == "Naive Bayes") {
      ggplot(data.frame(Actual = test_data$performance, Predicted = nb_predictions), 
             aes(x = Actual, fill = Predicted)) +
        geom_bar(position = "dodge") +
        labs(title = "Naive Bayes: Performance Distribution", 
             x = "Actual Performance", y = "Count", fill = "Predicted") +
        scale_fill_manual(values = c("Low" = "red", "Medium" = "orange", "High" = "green"))
    }
  })
}

# Run Shiny App
shinyApp(ui = ui, server = server)
