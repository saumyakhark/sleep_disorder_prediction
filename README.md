# sleep_disorder_prediction
Sleep Disorder Prediction using Machine Learning

This project aims to predict sleep disorders based on various health and lifestyle factors using machine learning. The model is built with a Random Forest Classifier and trained on a dataset containing information like sleep duration, physical activity, stress levels, heart rate, and more. The application is built using Flask and is deployed as a web service for user interaction.
Features:
  1) Predicts sleep disorder classification (No Sleep Disorder, Mild Sleep Disorder, Severe Sleep Disorder) based on user input.
  2) Provides personalized recommendations to improve sleep health for users with sleep disorders.
  3) User-friendly web interface with a form to collect health and lifestyle data.
Folder Structure:
/sleep-disorder-prediction
│
├── models/
│   └── sleep_disorder_model.pkl         # Trained model file
│
├── templates/
│   └── index.html                      # Main HTML file for form submission and displaying result
    |__ result.html
│
├── static/
│   └── style.css                        # CSS file for styling the webpage
│
├── app.py                               # Main Flask application
├── README.md                            # Project description and instructions

Installation Instructions
    1) Clone the repository: git clone https://github.com/saumyakhark/sleep-disorder-prediction.git
    2) Navigate to the project directory: cd sleep-disorder-prediction
    3) Create a virtual environment and activate it: python3 -m venv venv
source venv/bin/activate    # On Windows, use: venv\Scripts\activate

Usage
  1) Run the Flask Application: python app.py
  2) Open your browser and go to http://127.0.0.1:5000/ to interact with the form.
  3) Fill in the form with the required health and lifestyle data. The model will predict your sleep disorder status and provide helpful recommendations.

License
  This project is licensed under the MIT License - see the LICENSE file for details.
  
Dependencies
  1) Flask
  2) joblib
  3) scikit-learn
  4) numpy
