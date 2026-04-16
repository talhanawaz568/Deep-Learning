import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# --- Task 1: Generate Predictions ---
print("Task 1: Training Model and Generating Predictions...")

# 1.2 Load and Preprocess
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1.3 Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Generate predictions on the unseen test data
y_pred = clf.predict(X_test)

# --- Task 2: Create and Visualize Confusion Matrix ---
print("\nTask 2: Creating Confusion Matrix...")

# 2.1 Generate the Matrix
cm = confusion_matrix(y_test, y_pred)
print("Raw Confusion Matrix counts:\n", cm)

# 2.2 Visualization
plt.figure(figsize=(8, 6))
# 'annot=True' puts the numbers in the boxes; 'fmt=d' ensures they are integers
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix: Iris Species Classification')

# Save the plot for terminal-based environments
plt.savefig('confusion_matrix.png')
print("✓ Visualization saved as 'confusion_matrix.png'")

# --- Task 3: Interpret Results ---
print("\nTask 3: Detailed Classification Metrics")
print("-" * 40)
# This provides Precision, Recall, and F1-Score for each class
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report)


## pip install seaborn pandas matplotlib scikit-earn
