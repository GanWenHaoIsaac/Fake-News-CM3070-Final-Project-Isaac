from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import re
import nltk
import string
import shap
import pickle
import numpy as np
# import sys
# print(sys.executable)  # Should show the conda env path
# print(sys.path)        # Check if unwanted paths are prioritized#

# Load traditional ML model
tfidf_vectorizer = joblib.load("models/preprocessing_methods/tfidf_vectorizer_Notebook.pkl")
with open('models/preprocessing_methods/LSTMs-tokenizer.pkl', 'rb') as handle:
    lstm_tokenizer = pickle.load(handle)

with open('models/preprocessing_methods/BERTs-tokenizer.pkl', 'rb') as handle:
    bert_tokenizer = pickle.load(handle)

LR = joblib.load("models/logistic_regression_comp_4.pkl")
NB = joblib.load("models/naive_bayes_comp_4.pkl")
RF = joblib.load("models/random_forest_comp_2.pkl")
SVM = joblib.load("models/svm_model_comp_1.pkl")
DT = joblib.load("models/example_decision_tree.pkl")


from transformers import TFBertModel, TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input

LSTM = load_model("models/lstm_fixed.keras", compile=False)
CNN_LSTM = load_model("models/cnn_lstm_fixed.keras", compile=False)

BERT = load_model("models/bert_redo.keras",
                    custom_objects={'TFBertForSequenceClassification': TFBertForSequenceClassification}, 
                    compile=False)

BERT_LSTM = load_model("models/bert_LSTM_test_99.keras",
                        custom_objects={'TFBertModel': TFBertModel},
                        compile=False)

BIGRU = load_model("models/bigru_redo.keras", compile=False)

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], methods=["POST"], allow_headers=["Content-Type"])
import time


def bert_predict(text, model):
    """Handle BERT model predictions consistently"""
    try:
        # Tokenize input
        inputs = bert_tokenizer(
            text,
            return_tensors="tf",
            max_length=256,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False
        )
        
        # Get raw outputs
        outputs = model(inputs)
        
        # Handle different output formats
        if hasattr(outputs, 'logits'): 
            logits = outputs.logits.numpy()[0][0]
        elif isinstance(outputs, (np.ndarray, tf.Tensor)):
            logits = outputs[0][0] if isinstance(outputs, np.ndarray) else outputs.numpy()[0][0]
        else:
            raise ValueError("Unexpected model output format")
        
        # Convert to probability
        probability = 1 / (1 + np.exp(-logits))
        return probability
        
    except Exception as e:
        print(f"BERT Prediction Error: {str(e)}")
        raise


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
        #loading {
            display: none;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: #09f;
            animation: spin 1s linear infinite;
            margin: 10px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        #loading {
            display: none;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: #09f;
            animation: spin 1s linear infinite;
            margin: 10px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <h2>Fake News Detector</h2>
    <form action="/predict" method="post" id="prediction-form">
        <textarea name="text" rows="5" cols="50" placeholder="Enter news article...">{% if text %}{{ text }}{% endif %}</textarea><br><br>
        <label for="model">Choose a fake news detection model:</label>
        <select name="model" id="model">
            <option value="lr" {% if model_name == 'lr' %}selected{% endif %}>Logistic Regression</option>
            <option value="dt" {% if model_name == 'dt' %}selected{% endif %}>Decision Tree</option>
            <option value="svm" {% if model_name == 'svm' %}selected{% endif %}>SVM</option>
            <option value="nb" {% if model_name == 'nb' %}selected{% endif %}>Naive Bayes</option>
            <option value="rf"{% if model_name == 'rf' %}selected{% endif %}>Random Forest</option>
            <option value="lstm"{% if model_name == 'lstm' %}selected{% endif %}>LSTM (Long Short-Term Memory)</option>
            <option value="cnn-lstm"{% if model_name == 'cnn-lstm' %}selected{% endif %}>CNN+LSTM</option>
            <option value="bert"{% if model_name == 'bert' %}selected{% endif %}>BERT</option>
            <option value="bert-lstm"{% if model_name == 'bert-lstm' %}selected{% endif %}>BERT+LSTM</option>
            <option value="bigru"{% if model_name == 'bigru' %}selected{% endif %}>BiGRU</option>
        </select><br><br>
        <button type="submit">Check</button>
    </form>
    
    <div id="loading">
        <p>Analyzing article... Please wait...</p>
        <div class="spinner"></div>
    </div>
    
    
    <div id="loading">
        <p>Analyzing article... Please wait...</p>
        <div class="spinner"></div>
    </div>
    
    {% if prediction %}
    <h3>Model: {{ model_name|upper }}</h3>
    <h3> Prediction: {{ prediction }}</h3>
    <p class="confidence">Confidence: {{ confidence }}</p>
    {% endif %}

    <script>
        document.getElementById('prediction-form').addEventListener('submit', function() {
            document.getElementById('loading').style.display = 'block';
        });
        
        // Hide loader if prediction results are shown
        if(window.location.href.includes('/predict')) {
            document.getElementById('loading').style.display = 'none';
        }
    </script>

    <script>
        document.getElementById('prediction-form').addEventListener('submit', function() {
            document.getElementById('loading').style.display = 'block';
        });
        
        // Hide loader if prediction results are shown
        if(window.location.href.includes('/predict')) {
            document.getElementById('loading').style.display = 'none';
        }
    </script>
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
            model_name = request.form.get("model", "lr") 

        if not text:
            return render_template_string(HTML_FORM, 
                                          text=text, 
                                          model_name=model_name, 
                                          prediction="Please enter text.")


        # Deep learning models (LSTM/CNN-LSTM)
        if model_name in ["lstm", "cnn-lstm", "bigru"]:
            print(f"\nProcessing with {model_name.upper()} model...") 
            sequence = lstm_tokenizer.texts_to_sequences([text])
            if not sequence or not sequence[0]:
                print("Empty sequence after tokenization!")
                return render_template_string(HTML_FORM,
                                              text=text,
                                              model_name=model_name,
                                              prediction="Error: Text couldn't be processed")

            padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
            if model_name == "lstm":
                raw_pred = LSTM.predict(padded_sequence)
            elif model_name == "cnn-lstm":
                if CNN_LSTM is None:
                    print("Error: CNN_LSTM model is not loaded!")
                    return render_template_string(HTML_FORM, text=text, model_name=model_name, prediction="Error: Model not available")
                raw_pred = CNN_LSTM.predict(padded_sequence)
            elif model_name == "bigru":
                raw_pred = BIGRU.predict(padded_sequence)
            else:
                print(f"Error: Unknown model '{model_name}'")
                return render_template_string(HTML_FORM, text=text, model_name=model_name, prediction="Error: Invalid model name")
                
                
            print(f"Raw model output: {raw_pred}")  
                
            # Extract and validate prediction
            prediction = raw_pred[0][0]
            print(f"Final prediction value: {prediction}")            
            result = "Fake" if prediction < 0.5 else "Real"
            confidence = f"{max(prediction, 1-prediction)*100:.2f}%"


        elif model_name in ["bert", "bert-lstm"]:
            print(f"\nProcessing with {model_name.upper()} model...")
            try:

                tf.keras.backend.clear_session()
                model = BERT if model_name == "bert" else BERT_LSTM
                probability = bert_predict(text, model)
                # Determine result and confidence
                if probability < 0.5:
                    result = "Fake"
                    confidence = f"{(1 - probability)*100:.2f}%"
                else:
                    result = "Real"
                    confidence = f"{probability*100:.2f}%"
                    
            except Exception as e:
                predictionE=f"BERT Error: {str(e)}"
                print(predictionE)
                return render_template_string(HTML_FORM,
                                          text=text,
                                          model_name=model_name,
                                          prediction=predictionE)
            
        

        else:
            # Traditional ML models
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
                return render_template_string(HTML_FORM, prediction="Invalid model selected.") # Make a prediction
            
            text_vector = tfidf_vectorizer.transform([text])
            prediction = model.predict(text_vector)[0]
            proba = model.predict_proba(text_vector)[0]
            confidence = max(proba) * 100
            result = "Fake" if prediction == 0 else "Real"
        
        if isinstance(confidence, float):
            confidence_str = f"{confidence:.2f}%"
        else:
            confidence_str = confidence 

        return render_template_string(HTML_FORM, 
                                   text=text,
                                   model_name=model_name,
                                   prediction=result,
                                   confidence=confidence_str)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return render_template_string(HTML_FORM,
                                   text=text,
                                   model_name=model_name,
                                   prediction="Error occurred during prediction")
        
if __name__ == "__main__":
    app.run(debug=True)
