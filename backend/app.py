from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import re
import nltk
import string
import shap

# import sys
# print(sys.executable)  # Should show the conda env path
# print(sys.path)        # Check if unwanted paths are prioritized

# Load traditional ML model
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer_Notebook.pkl")

LR = joblib.load("models/logistic_regression_comp_4.pkl")
NB = joblib.load("models/naive_bayes_comp_4.pkl")
RF = joblib.load("models/random_forest_comp_2.pkl")
SVM = joblib.load("models/svm_model_comp_1.pkl")
DT = joblib.load("models/decision_tree.pkl")

# Preprocess text function
nltk.download("stopwords")
stop_words = set(nltk.corpus.stopwords.words("english"))


app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], methods=["POST"], allow_headers=["Content-Type"])

# HTML page for user input
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <style>
        .confidence {
            color: #666;
            font-style: italic;
            margin-top: -15px;
        }
    </style>
</head>
<body>
    <h2>Fake News Detector</h2>
    <form action="/predict" method="post">
        <textarea name="text" rows="5" cols="50" placeholder="Enter news article...">{% if text %}{{ text }}{% endif %}</textarea><br><br>
        <label for="model">Choose a model:</label>
        <select name="model" id="model">
            <option value="lr" {% if model_name == 'lr' %}selected{% endif %}>Logistic Regression</option>
            <option value="dt" {% if model_name == 'dt' %}selected{% endif %}>Decision Tree</option>
            <option value="svm" {% if model_name == 'svm' %}selected{% endif %}>SVM</option>
            <option value="nb" {% if model_name == 'nb' %}selected{% endif %}>Naive Bayes</option>
            <option value="rf"{% if model_name == 'rf' %}selected{% endif %}>Random Forest</option>
        </select><br><br>
        <button type="submit">Check</button>
    </form>
    {% if prediction %}
    <h3>Prediction: {{ prediction }}</h3>
    <p class="confidence">Confidence: {{ confidence }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_FORM)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({"message": "Use POST with JSON data to get a prediction."})

    try:
        text = ""
        model_name = "lr"
        if request.content_type == "application/json":
            data = request.get_json()
            text = data.get("text", "")
            model_name = data.get("model", "lr")  # Default to Logistic Regression
        
        else:
            text = request.form.get("text", "")
            model_name = request.form.get("model", "lr")  # Default to Logistic Regression

        if not text:
            return render_template_string(HTML_FORM, 
                                          text=text, 
                                          model_name=model_name, 
                                          prediction="Please enter text.")

        # Preprocess the text
        #processed_text = preprocess_text(text)

        # Vectorize the text
        text_vector = tfidf_vectorizer.transform([text])
        print(f"Received text: {text}")
        #print(f"Processed text: {processed_text}")


        if model_name == "lr":
            model = LR
        elif model_name == "nb":
            model = NB
        elif model_name == "dt":
            model = DT
        elif model_name == "svm":
            model = SVM
        elif model_name == "rf":
            model = RF
        else:
            return render_template_string(HTML_FORM, prediction="Invalid model selected.")

        # Make a prediction
        prediction = model.predict(text_vector)[0]
        proba = model.predict_proba(text_vector)[0]
        confidence = max(proba) * 100
        
        result = "Fake" if prediction == 0 else "Real"
        print(f"Model: {model}")
        print(f"Prediction (Flask): {prediction}")

        return render_template_string(HTML_FORM, 
                                   text=text,
                                   model_name=model_name,
                                   prediction=result,
                                   confidence=f"{confidence:.2f}%")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return render_template_string(HTML_FORM,
                                   text=text,
                                   model_name=model_name,
                                   prediction="Error occurred during prediction")
        
if __name__ == "__main__":
    app.run(debug=True)
