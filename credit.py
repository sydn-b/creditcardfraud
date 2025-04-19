import pandas as pd

df = pd.read_csv('creditcard.csv')  # Replace with your file path
print(df.head())


#STEP 1: Normalize 'Amount' and 'Time'
from sklearn.preprocessing import StandardScaler

# Create new normalized columns
df['Amount_Norm'] = StandardScaler().fit_transform(df[['Amount']])
df['Time_Norm'] = StandardScaler().fit_transform(df[['Time']])

# Drop the original 'Amount' and 'Time' columns
df = df.drop(['Amount', 'Time'], axis=1)


#STEP 2: Split Features and Target
# Separate features (X) and labels (y)
X = df.drop('Class', axis=1)
y = df['Class']



#STEP 3: Train-Test Split
from sklearn.model_selection import train_test_split

# Split into 80% training, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# STEP 4: Handle Class Imbalance with SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Optional: print to verify 
print("\n" + "="*60 + "\n")
print("Before SMOTE:", y_train.value_counts())

print("\n" + "="*60 + "\n")
print("After SMOTE:", pd.Series(y_train_resampled).value_counts())


#STEP 5: Train Your First Model (Logistic Regression)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# Predict
y_pred = model.predict(X_test)

# Evaluate 
print("\n" + "="*60 + "\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n" + "="*60 + "\n")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

#visualizing the Confusion Matrix

import seaborn as sns
import matplotlib.pyplot as plt

# Create a confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
