# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Sivavvp/engine-failure-prediction/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define target variable
target_col = 'Engine Condition'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

def cap_outliers(train, test, cols):
    train = train.copy()
    test = test.copy()

    for col in cols:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        train[col] = train[col].clip(lower, upper)
        test[col] = test[col].clip(lower, upper)

    return train, test

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define columns for outlier capping
cols_to_cap = ['Fuel pressure', 'Coolant pressure', 'lub oil temp']

# Apply capping
Xtrain, Xtest = cap_outliers(Xtrain, Xtest, cols_to_cap)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

print("Files saved successfully.")

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Sivavvp/engine-failure-prediction",
        repo_type="dataset",
    )
