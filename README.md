# 🏡 Boston House Price Prediction

An interactive machine learning web application that predicts the median value of owner-occupied homes in Boston. 

Live App: [View on Streamlit](https://shadowfox-2lw3zmrwx53u3q3hxwipcd.streamlit.app/)

## 📌 Project Overview
This repository contains a Linear Regression model built using `scikit-learn` to estimate housing prices. The app provides an interactive interface where users can adjust various urban and socio-economic factors to see how they impact property values in real-time.

## 📂 File Structure
 `app.py`: The Streamlit interface that handles user inputs and displays predictions.
 `BostonHousePrice.py`: The core data science script used for preprocessing, model training, and performance evaluation.
 `HousingData.csv`: The dataset containing 506 samples with features like crime rate, room count, and tax rates.
 `requirements.txt`: Lists the necessary Python packages (`pandas`, `scikit-learn`, `streamlit`, etc.) to run the environment.

## 📊 Dataset & Features
The model uses the following features to predict the MEDV (Median Value):
 RM: Average number of rooms per dwelling.
 LSTAT: Percentage of lower status of the population.
 CRIM: Per capita crime rate by town.
 PTRATIO: Pupil-teacher ratio by town.
 CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
 (And other socio-economic factors)

## 🛠️ Local Setup
To run this project on your machine:

1. Clone the repo:
   ```bash
   git clone [https://github.com/yashp1395/shadowfox.git](https://github.com/yashp1395/shadowfox.git)
