from flask import Flask, request, jsonify
import joblib
import re
import nltk

# Load traditional ML model
ml_model = joblib.load("models/logistic_regression.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Preprocess text function
nltk.download("stopwords")
stop_words = set(nltk.corpus.stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return ' '.join([word for word in text.split() if word not in stop_words])

app = Flask(__name__)

@app.route("/")
def home():
    return "Fake News Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    
    # Extract the text from the JSON data
    text = data.get('text', '')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Vectorize the text
    text_vector = tfidf_vectorizer.transform([processed_text])
    
    # Make a prediction
    prediction = ml_model.predict(text_vector)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)


    
# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import nltk
# import re
# from transformers import BertTokenizer, TFBertForSequenceClassification
# import tensorflow as tf

# # Load traditional ML model
# ml_model = joblib.load("models/logistic_regression.pkl")
# tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# # Load BERT model
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# # bert_model = TFBertForSequenceClassification.from_pretrained("models/bert_model/")

# # Preprocess text function
# nltk.download("stopwords")
# stop_words = set(nltk.corpus.stopwords.words("english"))

# def preprocess_text(text):
#     text = re.sub(r'\W', ' ', text)
#     text = text.lower()
#     text = re.sub(r'\s+', ' ', text)
#     return ' '.join([word for word in text.split() if word not in stop_words])

# app = Flask(__name__)

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     text = data.get("text", "")
#     model_type = data.get("model", "ml")  # Default to ML

#     processed_text = preprocess_text(text)

#     if model_type == "ml":
#         vectorized_text = tfidf_vectorizer.transform([processed_text])
#         prediction = ml_model.predict(vectorized_text)[0]
#     elif model_type == "bert":
#         tokens = tokenizer(processed_text, return_tensors="tf", max_length=512, truncation=True, padding="max_length")
#         logits = bert_model(tokens["input_ids"]).logits
#         prediction = tf.argmax(logits, axis=1).numpy()[0]

#     return jsonify({"prediction": "Real" if prediction == 1 else "Fake"})

# if __name__ == "__main__":
#     app.run(debug=True)
