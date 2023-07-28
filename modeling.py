import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score

from modeling_data import load_and_preprocess_data

# Load and preprocess the data
data = load_and_preprocess_data()

# Define the selected features
selected_features = ['risk_Risk 1 (High)', 'risk_Risk 2 (Medium)', 'risk_Risk 3 (Low)',
                     'inspection_season',
                     "Children's Services Facility", 'Grocery Store', 'Restaurant', 'School', 'Unknown Facility',
                     'Canvass', 'Complaint', 'Consultation', 'No Entry', 'Non-Inspection', 'Out Of Business', 'Recent Inspection',
                     'Suspected Food Poisoning', 'Tag Removal', 'Task Force', 'Unknown',
                     'violation_food', 'violation_facility', 'violation_sanitary', 'violation_staff', 'violation_unknown']

# Split the data into target variable (safety_score) and features
X = data[selected_features]
y = data['result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to calculate classification metrics
def calculate_classification_metrics(model, X, y_true, threshold=0.5):
    y_pred = model.predict(X)
    y_pred = (y_pred >= threshold).astype(int)  # Convert to binary predictions
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return mse, r2, accuracy, precision, recall, f1


# Define the hyperparameter grids for each model
random_forest_param_grid = {
    'n_estimators': [5, 10, 25],
    'max_depth': [2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10]
}

decision_tree_param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10]
}

knn_param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree'],
    'p': [1, 2]
}

# Perform GridSearchCV for each model
random_forest_gridsearch = GridSearchCV(RandomForestRegressor(), random_forest_param_grid, cv=5, scoring='neg_mean_squared_error')
decision_tree_gridsearch = GridSearchCV(DecisionTreeRegressor(), decision_tree_param_grid, cv=5, scoring='neg_mean_squared_error')
knn_gridsearch = GridSearchCV(KNeighborsRegressor(), knn_param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the models to the training data
random_forest_gridsearch.fit(X_train, y_train)
decision_tree_gridsearch.fit(X_train, y_train)
knn_gridsearch.fit(X_train, y_train)

# Perform cross-validation using the best estimator for each model
cv_scores_rf = cross_val_score(random_forest_gridsearch.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_scores_dt = cross_val_score(decision_tree_gridsearch.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_scores_knn = cross_val_score(knn_gridsearch.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Print the cross-validation scores for each model
print("Random Forest Cross-Validation Scores:", cv_scores_rf)
print("Decision Tree Cross-Validation Scores:", cv_scores_dt)
print("KNN Cross-Validation Scores:", cv_scores_knn)

# Print the mean cross-validation score for each model
print("Mean Random Forest CV Score:", cv_scores_rf.mean())
print("Mean Decision Tree CV Score:", cv_scores_dt.mean())
print("Mean KNN CV Score:", cv_scores_knn.mean())

# Get the best MSE for each model
best_rf_mse = -random_forest_gridsearch.best_score_
best_dt_mse = -decision_tree_gridsearch.best_score_
best_knn_mse = -knn_gridsearch.best_score_

# Print the best MSE for each model
print("Best Random Forest MSE:", best_rf_mse)
print("Best Decision Tree MSE:", best_dt_mse)
print("Best KNN MSE:", best_knn_mse)

# ... (Previous code remains the same)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Model 1: Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Model 2: Decision Tree Regression
    decision_tree_model = DecisionTreeRegressor(random_state=42)
    decision_tree_model.fit(X_train, y_train)

    # Model 3: Random Forest Regression
    random_forest_model = RandomForestRegressor(n_estimators=10, random_state=42)
    random_forest_model.fit(X_train, y_train)

    # Model 4: GradientBoost Classifier
    gradient_boosting_model = GradientBoostingClassifier()
    gradient_boosting_model.fit(X_train, y_train)

    # Model 5: Logistic Regression
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, y_train)

    # Model 6 KNN Regression model
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Model 7 Bagging Regression model
    bagging_model = BaggingRegressor(n_estimators=10, random_state=42)
    bagging_model.fit(X_train, y_train)

    # Calculate classification metrics for each model
    linear_mse, linear_r2, linear_accuracy, linear_precision, linear_recall, linear_f1 = calculate_classification_metrics(linear_model, X_test, y_test)
    decision_tree_mse, decision_tree_r2, decision_tree_accuracy, decision_tree_precision, decision_tree_recall, decision_tree_f1 = calculate_classification_metrics(decision_tree_model, X_test, y_test)
    random_forest_mse, random_forest_r2, random_forest_accuracy, random_forest_precision, random_forest_recall, random_forest_f1 = calculate_classification_metrics(random_forest_model, X_test, y_test)
    logistic_regression_mse, logistic_regression_r2, logistic_regression_accuracy, logistic_regression_precision, logistic_regression_recall, logistic_regression_f1 = calculate_classification_metrics(logistic_regression_model, X_test, y_test)
    knn_model_mse, knn_model_r2, knn_model_accuracy, knn_model_precision, knn_model_recall, knn_model_f1 = calculate_classification_metrics(knn_model, X_test, y_test)
    bagging_model_mse, bagging_model_r2, bagging_model_accuracy, bagging_model_precision, bagging_model_recall, bagging_model_f1 = calculate_classification_metrics(bagging_model, X_test, y_test)

    # Display the evaluation metrics for each model
    print("Linear Regression:")
    print(f"MSE: {linear_mse:.4f}, R-squared: {linear_r2:.4f}")
    print(f"Accuracy: {linear_accuracy:.4f}, Precision: {linear_precision:.4f}, Recall: {linear_recall:.4f}, F1 Score: {linear_f1:.4f}")
    print("")

    print("Decision Tree Regression:")
    print(f"MSE: {decision_tree_mse:.4f}, R-squared: {decision_tree_r2:.4f}")
    print(f"Accuracy: {decision_tree_accuracy:.4f}, Precision: {decision_tree_precision:.4f}, Recall: {decision_tree_recall:.4f}, F1 Score: {decision_tree_f1:.4f}")
    print("")

    print("Random Forest Regression:")
    print(f"MSE: {random_forest_mse:.4f}, R-squared: {random_forest_r2:.4f}")
    print(f"Accuracy: {random_forest_accuracy:.4f}, Precision: {random_forest_precision:.4f}, Recall: {random_forest_recall:.4f}, F1 Score: {random_forest_f1:.4f}")
    print("")

    print("Logistic Regression:")
    print(f"MSE: {logistic_regression_mse:.4f}, R-squared: {logistic_regression_r2:.4f}")
    print(f"Accuracy: {logistic_regression_accuracy:.4f}, Precision: {logistic_regression_precision:.4f}, Recall: {logistic_regression_recall:.4f}, F1 Score: {logistic_regression_f1:.4f}")
    print("")

    print("KNN Model:")
    print(f"MSE: {knn_model_mse:.4f}, R-squared: {knn_model_r2:.4f}")
    print(f"Accuracy: {knn_model_accuracy:.4f}, Precision: {knn_model_precision:.4f}, Recall: {knn_model_recall:.4f}, F1 Score: {knn_model_f1:.4f}")
    print("")

    print("Bagging Model:")
    print(f"MSE: {bagging_model_mse:.4f}, R-squared: {bagging_model_r2:.4f}")
    print(f"Accuracy: {bagging_model_accuracy:.4f}, Precision: {bagging_model_precision:.4f}, Recall: {bagging_model_recall:.4f}, F1 Score: {bagging_model_f1:.4f}")
    print("")

    # Define the hyperparameter grids for each model
    random_forest_param_grid = {
        'n_estimators': [5, 10, 25],
        'max_depth': [2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10]
    }

    decision_tree_param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [2, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10]
    }

    knn_param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree'],
        'p': [1, 2]
    }


    # Perform GridSearchCV for each model
    random_forest_gridsearch = GridSearchCV(RandomForestRegressor(), random_forest_param_grid, cv=5, scoring='neg_mean_squared_error')
    decision_tree_gridsearch = GridSearchCV(DecisionTreeRegressor(), decision_tree_param_grid, cv=5, scoring='neg_mean_squared_error')
    knn_gridsearch = GridSearchCV(KNeighborsRegressor(), knn_param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit the models to the training data
    random_forest_gridsearch.fit(X_train, y_train)
    decision_tree_gridsearch.fit(X_train, y_train)
    knn_gridsearch.fit(X_train, y_train)

    # Perform cross-validation using the best estimator for each model
    cv_scores_rf = cross_val_score(random_forest_gridsearch.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_scores_dt = cross_val_score(decision_tree_gridsearch.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_scores_knn = cross_val_score(knn_gridsearch.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    # Print the cross-validation scores for each model
    print("Random Forest Cross-Validation Scores:", cv_scores_rf)
    print("Decision Tree Cross-Validation Scores:", cv_scores_dt)
    print("KNN Cross-Validation Scores:", cv_scores_knn)

    # Print the mean cross-validation score for each model
    print("Mean Random Forest CV Score:", cv_scores_rf.mean())
    print("Mean Decision Tree CV Score:", cv_scores_dt.mean())
    print("Mean KNN CV Score:", cv_scores_knn.mean())

    # Get the best MSE for each model
    best_rf_mse = -random_forest_gridsearch.best_score_
    best_dt_mse = -decision_tree_gridsearch.best_score_
    best_knn_mse = -knn_gridsearch.best_score_

    # Print the best MSE for each model
    print("Best Random Forest MSE:", best_rf_mse)
    print("Best Decision Tree MSE:", best_dt_mse)
    print("Best KNN MSE:", best_knn_mse)

    # Return the best model and its metrics
    best_model = None
    best_model_mse = 0.0

    if best_rf_mse < best_model_mse:
        best_model = random_forest_gridsearch.best_estimator_
        best_model_mse = best_rf_mse

    if best_dt_mse < best_model_mse:
        best_model = decision_tree_gridsearch.best_estimator_
        best_model_mse = best_dt_mse

    if best_knn_mse < best_model_mse:
        best_model = knn_gridsearch.best_estimator_
        best_model_mse = best_knn_mse

    return best_model, best_model_mse



