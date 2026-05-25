import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder
if not os.path.exists("output"):
    os.makedirs("output")

# Load dataset
df = pd.read_csv("employee_data.csv")

print("\nDataset:")
print(df)

# Dataset info
print("\nDataset Info:")
print(df.info())

# Statistical summary
print("\nSummary:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# Remove duplicates
df = df.drop_duplicates()

# Correlation
correlation = df.select_dtypes(include=np.number).corr()

# ---------------- VISUALIZATIONS ----------------

sns.set(style="whitegrid")

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], kde=True, color="blue")
plt.title("Age Distribution")
plt.savefig("output/age_distribution.png")
plt.close()

# Salary Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Salary"], kde=True, color="green")
plt.title("Salary Distribution")
plt.savefig("output/salary_distribution.png")
plt.close()

# Department Count
plt.figure(figsize=(8,5))
sns.countplot(x="Department", data=df)
plt.title("Department Count")
plt.savefig("output/department_count.png")
plt.close()

# Experience vs Salary
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="Experience",
    y="Salary",
    hue="Department",
    data=df
)
plt.title("Experience vs Salary")
plt.savefig("output/experience_vs_salary.png")
plt.close()

# Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("output/correlation_heatmap.png")
plt.close()

print("\nEDA Completed Successfully!")