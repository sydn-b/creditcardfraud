import pandas as pd

df = pd.read_csv('creditcard.csv')  # Replace with your file path
print(df.head())

#STEP 1: Normalize 'Amount' and 'Time'
from sklearn.preprocessing import StandardScaler

#create new normalized columns
df['Amount_Norm'] = StandardScaler().fit_transform(df[['Amount']])
df['Time_Norm'] = StandardScaler().fit_transform(df[['Time']])

#drop the original 'Amount' and 'Time' columns
df = df.drop(['Amount', 'Time'], axis=1)


#STEP 2: Split Features and Target
#separate features (X) and labels (y)
X = df.drop('Class', axis=1)
y = df['Class']

#STEP 3: Train-Test Split
from sklearn.model_selection import train_test_split

#split into 80% training, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#STEP 4: Handle Class Imbalance with SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#print to verify 
print("\n" + "="*60 + "\n")
print("Before SMOTE:", y_train.value_counts())

print("\n" + "="*60 + "\n")
print("After SMOTE:", pd.Series(y_train_resampled).value_counts())


#STEP 5: Train Your First Model (Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#training the model
model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)

#gradient updates
model.fit(X_train_resampled, y_train_resampled)

#predict probabilities, this is the sigmoid function
y_probs = model.predict_proba(X_test)[:, 1]

#adjusting threshold value
threshold = 0.8 
y_pred = (y_probs >= threshold).astype(int)

#evaluate
print("\n" + "="*60 + "\n")
print(f"Evaluation with threshold = {threshold}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))


#visualizing the Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

#create a confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Threshold = {threshold})')
plt.tight_layout()
plt.show()
