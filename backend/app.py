from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import re
import nltk
import string
import shap

# Load traditional ML model
lr_model = joblib.load("models/logistic_regression.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
decision_tree_model = joblib.load("models/decision_tree.pkl")
svm_model = joblib.load("models/svm_model.pkl")

# Preprocess text function
nltk.download("stopwords")
stop_words = set(nltk.corpus.stopwords.words("english"))
# Initialize SHAP explainer
#explainer = shap.Explainer(lr_model, tfidf_vectorizer.transform)
import shap

masker = shap.maskers.Independent(tfidf_vectorizer.transform)
explainer = shap.LinearExplainer(lr_model, masker)

#explainer = shap.LinearExplainer(lr_model, tfidf_vectorizer, feature_perturbation="interventional")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?:\/\/\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], methods=["POST"], allow_headers=["Content-Type"])

# HTML page for user input
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <script>
        function showExplanation() {
            document.getElementById('explanation').style.display = 'block';
        }
    </script>
</head>
<body>
    <h2>Fake News Detector</h2>
    <form action="/predict" method="post">
        <textarea name="text" rows="5" cols="50" placeholder="Enter news article..."></textarea><br><br>
        <button type="submit">Check</button>
    </form>
    {% if prediction %}
    <h3>Prediction: {{ prediction }}</h3>
    <button onclick="showExplanation()">Why?</button>
    <div id="explanation" style="display: none;">
        <h4>Explanation:</h4>
        <p>{{ explanation }}</p>
    </div>
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
        if request.content_type == "application/json":
            text = request.get_json().get("text", "")
        else:
            text = request.form.get("text", "")

        if not text:
            return render_template_string(HTML_FORM, prediction="Please enter text.")

        # Preprocess the text
        processed_text = preprocess_text(text)

        # Vectorize the text
        text_vector = tfidf_vectorizer.transform([processed_text])

        # Make a prediction
        prediction = lr_model.predict(text_vector)
        result = "Fake" if prediction[0] == 1 else "Real"

        # Get SHAP explanation
        shap_values = explainer(text_vector)[0].values
        top_features = np.argsort(-np.abs(shap_values))[:3]
        feature_names = tfidf_vectorizer.get_feature_names_out()
        explanation = ", ".join(feature_names[i] for i in top_features)

        return render_template_string(HTML_FORM, prediction=result, explanation=explanation)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "Text is required"}), 400

        # Preprocess the text
        processed_text = preprocess_text(text)
        text_vector = tfidf_vectorizer.transform([processed_text])

        # Compute SHAP values
        shap_values = explainer(text_vector)[0].values
        top_features = np.argsort(-np.abs(shap_values))[:3]
        feature_names = tfidf_vectorizer.get_feature_names_out()
        explanation = {feature_names[i]: shap_values[i] for i in top_features}

        return jsonify({"explanation": explanation})
    except Exception as e:
        print(f"Error generating explanation: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
