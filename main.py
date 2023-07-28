# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from modeling import selected_features, best_knn_mse, best_dt_mse, decision_tree_gridsearch, random_forest_gridsearch, \
    knn_gridsearch, best_rf_mse


# Function to train and evaluate the models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # (Previous code remains the same)

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


if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('data_eda.csv')

    # ... (Rest of the code for data preprocessing remains the same)

    # Split the data into target variable (safety_score) and features
    X = data[selected_features]
    y = data['result']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate the models
    best_model, best_model_mse = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Print the best model and its metrics
    print("Best Model:", best_model)
    print("Best Model MSE:", best_model_mse)
