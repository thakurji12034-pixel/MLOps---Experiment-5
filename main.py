
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Create House Price Dataset
data = {
    "area": [1000, 1200, 1500, 1800, 2000,
             2200, 2500, 2800, 3000, 3500],
    "bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "age": [10, 8, 5, 4, 6, 3, 2, 5, 1, 2],
    "price": [2500000, 3000000, 4000000, 4500000, 5000000,
              5500000, 6500000, 7000000, 8000000, 9000000]
}

df = pd.DataFrame(data)

# 2. Separate Features and Target
X = df[["area", "bedrooms", "age"]]
y = df["price"]

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Machine Learning Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Predictions
predictions = model.predict(X_test)

# 6. Evaluate Model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("House Price Prediction Model")
print("--------------------------------")
print("Model Training Completed!")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# 7. Save Trained Model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/house_price_model.pkl")

print("Model saved successfully!")

# 8. Predict Price for a New House
new_house = pd.DataFrame({
    "area": [1800],
    "bedrooms": [3],
    "age": [4]
})

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])
