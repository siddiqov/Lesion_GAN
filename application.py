from flask import Flask, request, render_template, redirect, url_for
import os
from src.components.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)
predictor = PredictionPipeline()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)

        predicted_class = predictor.predict(filepath)
        return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
