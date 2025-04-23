import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('creditcard.csv')  #the file path to load the credit card transactions
print(df.head()) #preview the first five rows of the dataset

#normalize amount and time since all other features are PCA-transformed
df['Amount_Norm'] = StandardScaler().fit_transform(df[['Amount']])
df['Time_Norm'] = StandardScaler().fit_transform(df[['Time']])

#drop the original amount and time columns
df = df.drop(['Amount', 'Time'], axis=1)

#split dataset into features and labels
X = df.drop('Class', axis=1) #features
y = df['Class'] #labels (0 = Not Fraud, 1 = Fraud)

#split into 80% training, 20% test with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#print to verify class counts before and after SMOTE
print("\n============================================================\n")
print("Before SMOTE:", y_train.value_counts())

print("\n============================================================\n")
print("After SMOTE:", pd.Series(y_train_resampled).value_counts())

#training the logistic regression model with L1 regularization
model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)

#gradient updates
model.fit(X_train_resampled, y_train_resampled)

#predict probabilities using sigmoid activation
y_probs = model.predict_proba(X_test)[:, 1]

#applying a custom classification threshold value
threshold = 0.8 #a threshold of 0.8 improves precision over the default 0.5
y_pred = (y_probs >= threshold).astype(int)

#evaluate the performance using a confusion matrix and metrics
print("\n============================================================\n")
print(f"Evaluation with threshold = {threshold}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

#creating a confusion matrix heatmap visually as well
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Threshold = {threshold})')
plt.tight_layout()
plt.show()
