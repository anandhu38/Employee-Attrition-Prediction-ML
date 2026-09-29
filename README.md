

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

data = {
    'Age': [22, 35, 28, 45, 30, 50, 27, 41, 29, 55, 38, 32],
    'Salary': [25000, 60000, 35000, 85000, 40000, 90000,
               30000, 75000, 42000, 95000, 68000, 50000],
    'Experience': [1, 8, 3, 15, 4, 20, 2, 12, 5, 22, 10, 6],
    'Overtime': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'Attrition': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

print("\n===== DATASET =====")
print(df)



X = df[['Age', 'Salary', 'Experience', 'Overtime']]
y = df['Attrition']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\n===== MODEL PERFORMANCE =====")

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

