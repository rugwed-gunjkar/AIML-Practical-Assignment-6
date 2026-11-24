# AIML Assignment 6
# Basic Data Preprocessing using UCI Adult Income Dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Step 1: Load the dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = [
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
    "income"
]

df = pd.read_csv(
    url,
    names=columns,
    na_values="?",
    skipinitialspace=True
)

print("Original Dataset:")
print(df.head())

# Step 2: Check missing values

print("\nMissing Values Before Preprocessing:")
print(df.isnull().sum())

# Step 3: Handle missing values

numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include="object").columns

# Replace missing numerical values with median

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Replace missing categorical values with mode

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

# Step 4: Encode categorical variables

df = pd.get_dummies(
    df,
    columns=[col for col in categorical_cols if col != "income"],
    drop_first=True,
    dtype=int
)

# Encode target variable

le = LabelEncoder()
df["income"] = le.fit_transform(df["income"])

print("\nDataset After Encoding:")
print(df.head())

# Step 5: Normalize numerical features

scaler = MinMaxScaler()

numeric_features = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week"
]

df[numeric_features] = scaler.fit_transform(df[numeric_features])

print("\nDataset After Normalization:")
print(df[numeric_features].head())

# Step 6: Save processed dataset

df.to_csv("adult_income_processed.csv", index=False)

print("\nPreprocessed dataset saved successfully!")

print("\nFinal Dataset Shape:", df.shape)