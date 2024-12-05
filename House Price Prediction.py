import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate a dataset
np.random.seed(42)  # For reproducibility

# Generate features
num_samples = 100
house_sizes = np.random.randint(800, 3500, size=num_samples)  # in square feet
num_bedrooms = np.random.randint(1, 6, size=num_samples)  # number of bedrooms

# Simulate house prices with some noise
house_prices = (
    house_sizes * 150  # Base price per square foot
    + num_bedrooms * 10000  # Additional price per bedroom
    + np.random.normal(0, 50000, size=num_samples)  # Add noise
)

# Combine features into a dataset
X = np.column_stack((house_sizes, num_bedrooms))
y = house_prices

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Display model coefficients
print("Model Coefficients:")
print(f"  Price per square foot: {model.coef_[0]:.2f}")
print(f"  Price per bedroom: {model.coef_[1]:.2f}")
print(f"  Intercept: {model.intercept_:.2f}")

# Visualize the predictions
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual Prices")
plt.scatter(range(len(y_pred)), y_pred, color="red", label="Predicted Prices")
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("House Price")
plt.title("Actual vs. Predicted House Prices")
plt.show()
