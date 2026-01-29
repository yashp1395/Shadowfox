import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath='HousingData.csv'):
    """Load data and handle missing values using median imputer."""
    data = pd.read_csv(filepath)
    
    # Handle missing values - replace NaN with the median of each column
    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    return data_imputed

def train_boston_model(data):
    """Split data and train a Linear Regression model."""
    # Features and target variable
    X = data.drop(columns=['MEDV'])
    y = data['MEDV']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def main():
    # Load and Train
    data = load_and_preprocess_data()
    model, X_test, y_test = train_boston_model(data)

    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.show()

if __name__ == "__main__":
    main()
