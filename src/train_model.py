import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load the dataset
data_path = '../datasets/cancer_risk_data.csv'
data = pd.read_csv(data_path)

# Define the features and target
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

# Save the model and scaler
joblib.dump(model, 'models/gradient_boosting_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
