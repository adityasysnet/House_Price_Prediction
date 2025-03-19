import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset (assuming housing_data is a DataFrame with relevant columns)
housing_data = pd.read_csv('housing.csv')
# Convert categorical variables to numeric values
housing_data['mainroad'] = housing_data['mainroad'].map({'yes': 1, 'no': 0})
housing_data['guestroom'] = housing_data['guestroom'].map({'yes': 1, 'no': 0})
housing_data['basement'] = housing_data['basement'].map({'yes': 1, 'no': 0})
housing_data['hotwaterheating'] = housing_data['hotwaterheating'].map({'yes': 1, 'no': 0})
housing_data['airconditioning'] = housing_data['airconditioning'].map({'yes': 1, 'no': 0})
housing_data['prefarea'] = housing_data['prefarea'].map({'yes': 1, 'no': 0})
housing_data['furnishingstatus'] = housing_data['furnishingstatus'].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})

X = housing_data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']].values  # Multiple features
y = housing_data['price'].values  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("House Price Prediction App")
st.write("Enter house details to predict the price.")

# User Inputs
area = st.number_input("Enter the area (sq ft)", min_value=100, max_value=10000, value=1500)
bedrooms = st.number_input("Enter the number of bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Enter the number of bathrooms", min_value=1, max_value=5, value=2)
stories = st.number_input("Enter the number of stories", min_value=1, max_value=5, value=1)
mainroad = st.radio("Is it on the main road?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
guestroom = st.radio("Does it have a guestroom?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
basement = st.radio("Does it have a basement?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
hotwaterheating = st.radio("Does it have hot water heating?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
airconditioning = st.radio("Does it have air conditioning?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
parking = st.number_input("Number of parking spaces", min_value=0, max_value=5, value=1)
prefarea = st.radio("Is it in a preferred area?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
furnishingstatus = st.radio("Furnishing status", [0, 1, 2], format_func=lambda x: ['Unfurnished', 'Semi-furnished', 'Furnished'][x])

pred_price = None  # Initialize pred_price

# Predict Button
if st.button("Predict Price"):
    user_input = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]])
    pred_price = model.predict(user_input)
    st.success(f"Predicted price: ${pred_price[0]:,.2f}")

# Visualizing (only possible for one variable at a time, e.g., bedrooms vs price)
st.subheader("Visualization")
fig, ax = plt.subplots()
ax.scatter(housing_data['bedrooms'], housing_data['price'], color='blue', label='Actual Prices')
if pred_price is not None:
    ax.scatter(bedrooms, pred_price, color='red', marker='x', label='Predicted Price')
ax.set_xlabel('Number of Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Price vs Number of Bedrooms')
ax.legend()
st.pyplot(fig)
