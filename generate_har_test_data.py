import pandas as pd

features = pd.read_csv("UCI HAR Dataset/features.txt", delim_whitespace=True, header=None)[1].values

# Fix duplicates
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

features = make_unique(features)

X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", delim_whitespace=True, header=None)
X_test.columns = features

# Save first 5 rows
X_test.head(5).to_csv("sample_input.csv", index=False)
