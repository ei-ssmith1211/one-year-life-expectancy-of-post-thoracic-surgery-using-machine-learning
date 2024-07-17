from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your model
model = pickle.load(open('best_random_forest_model.pkl', 'rb'))

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        form_data = request.form.to_dict()
        # Convert form data to numpy array
        input_features = np.array([[float(form_data['FVC']),
                                    float(form_data['FEV1']),
                                    int(form_data['Performance']),
                                    int(form_data['Pain']),
                                    int(form_data['Haemoptysis']),
                                    int(form_data['Dyspnoea']),
                                    int(form_data['Cough']),
                                    int(form_data['Weakness']),
                                    float(form_data['Tumor_Size']),
                                    int(form_data['Diabetes_Mellitus']),
                                    int(form_data['MI_6mo']),
                                    int(form_data['PAD']),
                                    int(form_data['Smoking']),
                                    int(form_data['Asthma']),
                                    int(form_data['Age'])]])

        # Normalize the input features
        input_features_normalized = scaler.transform(input_features)

        # Make prediction with the loaded model
        prediction = model.predict(input_features_normalized)
        
        # Determine the prediction result
        prediction_text = 'Patient is at High Risk' if prediction[0] == 0 else 'Patient is Not at Risk'

        return render_template('predict.html', prediction_text=f'The Prediction is: {"Patient is at High Risk" if prediction[0] == 0 else "Patient is Not at Risk"}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
