import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, 'exercise_detection', 'exercise_angles.csv')
df = pd.read_csv(csv_path)

label_counts = df['Label'].value_counts()
print("1) Label distribution:")
print(label_counts, "\n")
plt.figure(figsize=(6,4))
label_counts.plot(kind='bar')
plt.title('Exercise Label Distribution')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


numeric = df.select_dtypes(include=[np.number])
print("2) Numeric feature summary (mean, std, min, max):")
print(numeric.describe().T, "\n")

for col in numeric.columns:
    plt.figure(figsize=(5,3))
    plt.hist(numeric[col].dropna(), bins=40)
    plt.title(f'{col} distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

corr = numeric.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, cmap='coolwarm', center=0, square=True)
plt.title('4) Feature Correlation Matrix')
plt.tight_layout()
plt.show()

print("5) Detailed histograms for first 6 features")
for col in numeric.columns[:6]:
    plt.figure(figsize=(5,3))
    plt.hist(numeric[col].dropna(), bins=30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
