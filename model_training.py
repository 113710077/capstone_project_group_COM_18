import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Show how many NaNs are in CLASS
print("NaNs in CLASS column:", df["CLASS"].isna().sum())

# Remove rows where CLASS is NaN
df = df.dropna(subset=["CLASS"])

# Convert CLASS to int (sometimes it becomes float after dropna)
df["CLASS"] = df["CLASS"].astype(int)

# Features used for training
features = ["API_MIN", "API", "vt_detection", "VT_Malware_Deteccao", "AZ_Malware_Deteccao"]

# Remove rows where any feature is missing
df = df.dropna(subset=features)

X = df[features]
y = df["CLASS"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training complete! Model saved as rf_model.pkl")
