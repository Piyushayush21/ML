import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create some dummy data
# Let's say X is a feature (e.g., square footage), and y is the target (e.g., price)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + Gaussian noise
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = model.predict(X_test)

# 5. Evaluate the model
print("--- Linear Regression Results ---")
print(f"Coefficients (Slope): {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Mean squared error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"Coefficient of determination (R^2): {r2_score(y_test, y_pred):.2f}")

# 6. Plot the results and save as image
plt.scatter(X_test, y_test, color='black', label='Actual Data (Test Set)')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Regression Line')
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Simple Linear Regression')
plt.legend()
plt.savefig('linear_regression_plot.png')
print("\nPlot saved as 'linear_regression_plot.png'")
