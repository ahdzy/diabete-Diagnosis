##
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

# Load the model and dataset
model = joblib.load('model.joblib')
csv_file = "diabete.csv"
df = pd.read_csv(csv_file)

# Define the independent variables (X) and the dependent variable (y)
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = df['Outcome']

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Create a Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
area_under_curve = auc(recall, precision)

plt.figure(figsize=(8, 8))
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {area_under_curve:.2f})', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()
