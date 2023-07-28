# captsone_project


# Chicago Food Inspections Data Analysis

This repository contains code and data analysis for Chicago food inspections. The goal of this project is to analyze food inspection data, explore patterns, and provide insights into the safety scores and violations in various restaurants.


## Table of Contents

1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Model Selection](#model-selection)
5. [Model Evaluation](#model-evaluation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Conclusion](#conclusion)


## Data Analysis Libraries

The code uses the following libraries for data analysis and visualization:

- Pandas: Used for data manipulation and analysis.
- NumPy: Utilized for numerical operations and calculations.
- Matplotlib and Seaborn: Used for data visualization and creating plots.
- WordCloud: Used for creating a word cloud visualization of the most common words in restaurant names.
- Geopandas and Folium: Used for geospatial data visualization.

## Installation and Data Loading

To run the code, make sure you have the required libraries installed. You can install them using the following command:

```bash
pip install pandas numpy matplotlib seaborn wordcloud geopandas folium
```

The data is loaded from the Chicago government's public data API. It is a CSV file containing information about food inspections in the city of Chicago. The data is retrieved using the 'requests' library and then loaded into a Pandas DataFrame for further analysis.

## Data Preprocessing

The code performs data preprocessing tasks to clean and prepare the data for analysis. Some of the preprocessing steps include:

- Handling missing values in various columns.
- Transforming the 'inspection_date' column to a datetime format.
- Categorizing and encoding violation codes into generalized categories for better analysis.
- Grouping facility types and risks for analysis.

## Exploratory Data Analysis (EDA)

The EDA section analyzes the data and creates various visualizations to gain insights into the dataset. Some of the visualizations include:

- Distribution of inspections over the years and risk levels.
- Violations distribution for passed and failed inspections.
- Mean safety scores over the years for top facility types and zip codes.

## Geospatial Data Visualization

The code utilizes Folium to create interactive maps for visualizing the distribution of food facilities and safety scores in Chicago. The maps are color-coded based on safety scores and facility locations.

## Model Evaluation and Hyperparameter Tuning

After training multiple regression and classification models, the code proceeds with evaluating their performance and tuning the hyperparameters for better results. Here's a summary of what's happening in this part of the code:

### Model Evaluation
1. The classification metrics for each model (Linear Regression, Decision Tree Regression, Random Forest Regression, Logistic Regression, KNN Regression, and Bagging Regression) are calculated using the `calculate_classification_metrics` function.
2. The calculated metrics include Mean Squared Error (MSE), R-squared, Accuracy, Precision, Recall, and F1 Score.
3. The evaluation metrics for each model are then displayed.

### Hyperparameter Tuning
1. Hyperparameter grids are defined for each model (Random Forest, Decision Tree, and KNN) to search for the best hyperparameters.
2. The `GridSearchCV` function from scikit-learn is used to perform hyperparameter tuning using cross-validation.
3. The models are refitted to the training data using the best hyperparameters.
4. Cross-validation is performed using the best estimator for each model to further validate their performance.
5. The mean cross-validation score and best Mean Squared Error (MSE) for each model are displayed.

Please note that hyperparameter tuning is an essential step in machine learning to optimize the performance of the models and prevent overfitting. The GridSearchCV function exhaustively searches the specified hyperparameter space to find the best combination of hyperparameters for each model.

It is important to analyze the evaluation metrics and cross-validation scores for each model to determine which model performs best for the specific task at hand. These scores can help in selecting the most suitable model for predicting the target variable, in this case, whether an inspection result is a 'Pass' or 'Fail'.

Keep in mind that the success of hyperparameter tuning depends on the size and quality of the dataset, as well as the relevance of the chosen hyperparameter space. You can adjust the hyperparameter grids based on domain knowledge and prior experience with similar datasets and models.

Make sure to interpret the results carefully and consider the specific requirements and constraints of your project when choosing the final model for deployment.


## Conclusion

In this code, we have performed an extensive analysis and modeling of the dataset to predict inspection outcomes as 'Pass' or 'Fail' based on various features. Here are the key takeaways from our analysis:

1. Data Preprocessing: We started by loading the dataset and performing data preprocessing steps, such as one-hot encoding categorical variables like 'inspection_type' and 'facility_type_grouped.' We also converted the 'risk' column into dummy variables to make it suitable for modeling.

2. Feature Engineering: To gain deeper insights, we created a function to count the number of violations for each inspection record based on the provided violation codes. This allowed us to extract valuable information and create new features related to different types of violations.

3. Model Selection: We trained multiple regression and classification models, including Linear Regression, Decision Tree Regression, Random Forest Regression, Logistic Regression, KNN Regression, and Bagging Regression. Each model was evaluated based on various classification metrics like Accuracy, Precision, Recall, and F1 Score.

4. Model Evaluation: Our evaluation metrics provided a comprehensive overview of each model's performance. We considered not only the predictive accuracy but also the precision and recall, which are crucial when dealing with imbalanced classes like 'Pass' and 'Fail.' These metrics help us assess the models' effectiveness in identifying failed inspections correctly.

5. Hyperparameter Tuning: To further optimize our models, we performed hyperparameter tuning using GridSearchCV with cross-validation. This helped us find the best combination of hyperparameters for Random Forest, Decision Tree, and KNN models.

6. Recommendations: Based on the evaluation metrics and cross-validation scores, the **Random Forest Regression model** showed the most promising performance in predicting inspection outcomes. However, it is essential to consider the specific context and requirements of the application before finalizing the model for deployment.

7. Deployment: Once the final model is selected, it can be deployed into a production environment for real-time inspection outcome predictions. Additionally, it's essential to monitor the model's performance regularly and retrain it with updated data to ensure its accuracy and effectiveness over time.

In summary, our analysis and modeling efforts provide valuable insights into predicting inspection outcomes. The code presented here can serve as a strong foundation for building a robust and reliable system to support food safety inspections and facilitate better decision-making for regulatory authorities and businesses alike.