import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv("noma_ofdm_dataset.csv", header=None)

X = data.iloc[:, 0:2]  # SNR, users
Y = data.iloc[:, 2]    # rate difference

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Model
model = RandomForestRegressor(n_estimators=100)

# Train
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)
print("Model Accuracy:", score)

# Save model
joblib.dump(model, "noma_ofdm_model.pkl")

print("Model saved!")