import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load data
data_path = os.path.join("data", "salary_data.csv")
df = pd.read_csv(data_path)

X = df[['YearsExperience']]
y = df['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
with open(os.path.join("model", "salary_model.pkl"), "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully!")
