import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the processed data
data = pd.read_csv('data/processed_data.csv')  # Ensure this path matches where your preprocessed data is saved

# Split data into features (X) and target (y)
X = data.drop(columns=['Person ID', 'Sleep Disorder'])  # Drop 'Person ID' and 'Sleep Disorder' as target
y = data['Sleep Disorder']  # The target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'models/sleep_disorder_model.pkl')
