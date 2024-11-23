from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('models/sleep_disorder_model.pkl')

# Route to display the form for prediction
@app.route('/')
def index():
    return render_template('index.html')  # HTML form for input

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from the form
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    sleep_duration = float(request.form['sleep_duration'])
    physical_activity = float(request.form['physical_activity'])
    stress_level = float(request.form['stress_level'])
    bmi_category = int(request.form['bmi_category'])
    systolic_bp = int(request.form['systolic_bp'])
    diastolic_bp = int(request.form['diastolic_bp'])
    heart_rate = int(request.form['heart_rate'])
    daily_steps = int(request.form['daily_steps'])
    smoking = int(request.form['smoking'])
    alcohol_consumption = int(request.form['alcohol_consumption'])

    # Prepare input for the model
    input_features = np.array([[age, gender, sleep_duration, physical_activity, stress_level,
                                bmi_category, systolic_bp, diastolic_bp, heart_rate, daily_steps,
                                smoking, alcohol_consumption]])
    
    # Predict sleep disorder class
    prediction = model.predict(input_features)

    # Convert prediction to readable form
    if prediction == 0:
        result = "No Sleep Disorder"
        steps = "You're in good health! Keep maintaining a balanced lifestyle."
    elif prediction == 1:
        result = "Mild Sleep Disorder"
        steps = "You should aim to improve your sleep hygiene, reduce stress, and engage in regular physical activity."
    else:
        result = "Severe Sleep Disorder"
        steps = "It's highly recommended to consult a healthcare professional for a personalized plan to manage your sleep disorder."
    
    # Return the result and steps on a new page
    return render_template('result.html', prediction=result, steps=steps)

if __name__ == '__main__':
    app.run(debug=True)
