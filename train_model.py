import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import joblib

# Load the diabetes dataset
diabetes = load_diabetes()
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
target = diabetes.target

# Split data into features and target variable
X = data
y = target

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, "model.joblib")

print("Model successfully trained and saved to model.joblib")

raise Exception("General exception")
