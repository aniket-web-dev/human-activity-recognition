import pandas as pd

# Load features
features = pd.read_csv("UCI HAR Dataset/features.txt", delim_whitespace=True, header=None)[1].values

# Ensure unique columns
def make_unique(cols):
    seen = {}
    result = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            result.append(col)
    return result

features = make_unique(features)

# Load data
X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", delim_whitespace=True, header=None)
X_test.columns = features

y_test = pd.read_csv("UCI HAR Dataset/test/y_test.txt", delim_whitespace=True, header=None)

# Combine first 3 examples from each label (1–6)
diverse_samples = pd.DataFrame()
for label in range(1, 7):
    indices = y_test[y_test[0] == label].index[:3]
    diverse_samples = pd.concat([diverse_samples, X_test.loc[indices]])

# Save to CSV
diverse_samples.to_csv("diverse_input.csv", index=False)
