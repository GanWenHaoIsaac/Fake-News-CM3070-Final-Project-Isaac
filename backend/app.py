from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import numpy as np
from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import deque

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load Models
tfidf_vectorizer = joblib.load("models/preprocessing_methods/tfidf_vectorizer_Notebook.pkl")
lstm_tokenizer = pickle.load(open('models/preprocessing_methods/LSTMs-tokenizer.pkl', 'rb'))
bert_tokenizer = pickle.load(open('models/preprocessing_methods/BERTs-tokenizer.pkl', 'rb'))

models = {
    "lr": joblib.load("models/logistic_regression_comp_4.pkl"),
    "nb": joblib.load("models/naive_bayes_comp_4.pkl"),
    "rf": joblib.load("models/random_forest_comp_2.pkl"),
    "svm": joblib.load("models/svm_model_comp_1.pkl"),
    "dt": joblib.load("models/decision_tree.pkl"),
    "lstm": load_model("models/lstm_fixed.keras", compile=False),
    "cnn-lstm": load_model("models/cnn_lstm_fixed.keras", compile=False),
    "bert": load_model("models/bert_redo.keras", custom_objects={'TFBertForSequenceClassification': TFBertForSequenceClassification}, compile=False),
    "bert-lstm": load_model("models/bert_LSTM_test_99.keras", custom_objects={'TFBertModel': TFBertModel}, compile=False),
    "bigru": load_model("models/bigru_redo.keras", compile=False)
}

# Store last 5 predictions in history
history = deque(maxlen=5)

def bert_predict(text, model):
    try:
        inputs = bert_tokenizer(text, return_tensors="tf", max_length=256, padding='max_length', truncation=True, return_token_type_ids=False)
        outputs = model(inputs)
        logits = outputs.logits.numpy()[0][0] if hasattr(outputs, 'logits') else outputs.numpy()[0][0]
        probability = 1 / (1 + np.exp(-logits))  # Applied sigmoid
        return probability
    except Exception as e:
        print(f"BERT Prediction Error: {str(e)}")
        return None

@app.route("/", methods=["GET", "POST"])
def home():
    return predict()

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({"message": "Use POST with JSON data to get a prediction."})
    
    try:
        data = request.get_json()
        text = data.get("text", "")
        model_name = data.get("model", "lr").lower()

        if not text:
            return jsonify({"error": "No text provided."}), 200

        if model_name not in models:
            return jsonify({"error": "Invalid model selected."}), 400

        # Prediction logic - Deep Learning Models
        model = models[model_name]
        if model_name in ["lstm", "cnn-lstm"]:
            sequence = lstm_tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
            probability = model.predict(padded_sequence)[0][0]

        elif model_name in ["bert", "bert-lstm"]:
            probability = bert_predict(text, model)
            if probability is None:
                return jsonify({"error": "BERT model prediction failed."}), 500
            
        else:  # Traditional ML models
            text_vector = tfidf_vectorizer.transform([text])         
            probability = model.predict_proba(text_vector)[0][1]

        result = "Real" if probability >= 0.5 else "Fake"
        confidence = f"{probability * 100:.2f}%" if result == "Real" else f"{(1 - probability) * 100:.2f}%"

        # Store prediction in history
        history.appendleft({
            "text": text,
            "model": model_name,
            "prediction": result,
            "confidence": confidence
        })


        return jsonify({"model": model_name, "prediction": result, "confidence": confidence})
    
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({"error": "An error occurred during prediction."}), 500

@app.route("/history", methods=["GET"])
def get_history():
    return jsonify({"history": list(history)})

if __name__ == "__main__":
    app.run(debug=True)
