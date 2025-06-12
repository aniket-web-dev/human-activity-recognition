import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 📌 Load and fix duplicate feature names
features = pd.read_csv("UCI HAR Dataset/features.txt", delim_whitespace=True, header=None)
feature_names = features[1].values

def make_unique(cols):
    seen = {}
    unique_cols = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            unique_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            unique_cols.append(col)
    return unique_cols

# Apply fix
feature_names = make_unique(feature_names)

# 📥 Load training data
X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", delim_whitespace=True, header=None)
y_train = pd.read_csv("UCI HAR Dataset/train/y_train.txt", delim_whitespace=True, header=None).values.ravel()

# 📥 Load test data (optional)
X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", delim_whitespace=True, header=None)
y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", delim_whitespace=True, header=None).values.ravel()

# 🏷️ Set fixed feature names
X_train.columns = feature_names
X_test.columns = feature_names

# 🤖 Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 💾 Save model
joblib.dump((model, feature_names), "har_model.pkl")
print("✅ Model trained and saved with feature names.")