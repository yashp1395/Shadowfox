import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title
st.set_page_config(page_title="Boston House Price Predictor", layout="wide")
st.title("🏡 Boston House Price Prediction App")

# 1. Load and Prepare Data (Replicating logic from BostonHousePrice.py)
@st.cache_resource
def train_model():
    # Load dataset
    data = pd.read_csv('HousingData.csv')
    
    # Handle missing values using median strategy
    imputer = SimpleImputer(strategy='median')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Features and target
    X = data_imputed.drop(columns=['MEDV'])
    y = data_imputed['MEDV']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X, data_imputed

model, X_features, df_full = train_model()

# 2. Sidebar for User Input
st.sidebar.header("User Input Features")

def get_user_inputs():
    inputs = {}
    # Dynamically create sliders based on dataset feature ranges
    for col in X_features.columns:
        min_val = float(df_full[col].min())
        max_val = float(df_full[col].max())
        mean_val = float(df_full[col].mean())
        inputs[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)
    
    return pd.DataFrame(inputs, index=[0])

input_df = get_user_inputs()

# 3. Main Interface Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Parameters")
    st.write(input_df)
    
    if st.button("Predict Price"):
        # Make prediction based on slider values
        prediction = model.predict(input_df)
        st.success(f"### Predicted Median House Value (MEDV): ${prediction[0]:.2f}k")

with col2:
    st.subheader("Model Visualization")
    # Replicating the Actual vs Predicted plot logic
    y_test_pred = model.predict(X_features)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df_full['MEDV'], y=y_test_pred, color='blue', ax=ax)
    plt.plot([df_full['MEDV'].min(), df_full['MEDV'].max()], 
             [df_full['MEDV'].min(), df_full['MEDV'].max()], color='red', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices (Full Dataset)')
    st.pyplot(fig)

# 4. Data Summary
if st.checkbox("Show Raw Data Summary"):
    st.write(df_full.describe())
