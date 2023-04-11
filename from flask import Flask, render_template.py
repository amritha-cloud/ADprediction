from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('alzheimers.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    education = int(request.form['education'])
    ses = int(request.form['ses'])
    gender = int(request.form['gender'])
    apoe4 = int(request.form['apoe4'])

    # Preprocess the input data
    input_data = np.array([[age, education, ses, gender, apoe4]])
    input_data = (input_data - np.mean(input_data)) / np.std(input_data)

    # Make a prediction using the trained model
    prediction = model.predict(input_data)[0][0]

    # Convert the prediction to a string
    if prediction > 0.5:
        output = 'Alzheimer\'s Disease'
    else:
        output = 'No Alzheimer\'s Disease'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)