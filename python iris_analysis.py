# Assignment: Analyzing Data with Pandas and Visualizing Results with Matplotlib
# Dataset: Iris Dataset (from sklearn)
# This script performs data loading, exploration, basic analysis, and visualizations
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris()

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
print(df.head())

# Check data types and missing values
print(df.info())
print(df.isnull().sum())  # Check for missing values
# Summary statistics of numerical columns
print(df.describe())

# Group by the target (species) and compute the mean of numerical columns
grouped = df.groupby('target').mean()
print(grouped)

import matplotlib.pyplot as plt

# Plot the trend of sepal length (even though no time series data, we can visualize the values)
plt.plot(df['sepal length (cm)'])
plt.title("Trend of Sepal Length")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.show()

# Bar chart showing average petal length per species
plt.bar(df['target'].unique(), df.groupby('target')['petal length (cm)'].mean())
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# Histogram to show the distribution of sepal length
df['sepal length (cm)'].hist(bins=20)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot to visualize the relationship between sepal length and petal length
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

try:
    df = pd.read_csv('iris.csv')  # This line is not used for Iris dataset, but useful for other datasets
except FileNotFoundError:
    print("File not found. Please check the file path.")

# Conclusion:
# - The average petal length varies significantly across the species.
# - The distribution of sepal lengths is somewhat uniform.
# - The scatter plot shows a positive correlation between sepal length and petal length.