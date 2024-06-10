import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Read data from CSV file
csv_file = "diabete.csv"
df = pd.read_csv(csv_file)

# Define the independent variables (X) and the dependent variable (y)
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = df['Outcome']

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include=['number']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Create a transformer to handle numeric and categorical columns separately
numeric_transformer = Pipeline(steps=[
    ('num', 'passthrough')  # No transformation for numeric columns
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))  # One-hot encode categorical columns
])

# Combine transformers into a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Create a random forest classifier model with the preprocessor
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model for classification
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the classification metrics
print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)

# Save the model to a file
joblib.dump(model, 'model.joblib')
