import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the raw dataset
data = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')

# Check the column names to avoid KeyError
print("Columns in the dataset:", data.columns)

# Handle missing data (filling with mean for numerical and mode for categorical)
numeric_data = data.select_dtypes(include=['float64', 'int64'])
data[numeric_data.columns] = numeric_data.fillna(numeric_data.mean())  # Fill missing numerical values with mean
categorical_columns = data.select_dtypes(include=['object']).columns  # Identify categorical columns
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)  # Fill missing categorical values with mode

# Convert Blood Pressure from 'systolic/diastolic' to separate columns
if 'Blood Pressure' in data.columns:
    # Split the 'Blood Pressure' column
    blood_pressure_split = data['Blood Pressure'].str.split('/', expand=True)
    data['Systolic BP'] = pd.to_numeric(blood_pressure_split[0], errors='coerce')
    data['Diastolic BP'] = pd.to_numeric(blood_pressure_split[1], errors='coerce')
    data.drop(columns=['Blood Pressure'], inplace=True)  # Drop the original 'Blood Pressure' column

# Encode categorical variables (e.g., Gender, Occupation, Physical Activity Level, BMI Category)
label_encoder = LabelEncoder()

# Encode the 'Gender' column
if 'Gender' in data.columns:
    data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Encoding other categorical columns
categorical_columns = ['Occupation', 'Physical Activity Level', 'BMI Category', 'Sleep Disorder']
for col in categorical_columns:
    if col in data.columns:
        data[col] = label_encoder.fit_transform(data[col])

# Update numerical columns list based on dataset inspection
numerical_columns = ['Age', 'Sleep Duration', 'Stress Level', 'Systolic BP', 'Diastolic BP', 'Heart Rate', 'Daily Steps']
data[numerical_columns] = StandardScaler().fit_transform(data[numerical_columns])

# Save processed data
data.to_csv('data/processed_data.csv', index=False)

# Optional: Generate a missing data report
missing_data_report = data.isnull().sum()
missing_data_report.to_csv('data/missing_data_report.csv')

# Display the processed data
print(data.head())
