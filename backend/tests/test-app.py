import sys
import os
sys.path.insert(0, "Users\isaac\Desktop\CM3070-FakeNews-Vite-App\my-fake-news-app\backend\models")
sys.path.insert(0, "Users\isaac\Desktop\CM3070-FakeNews-Vite-App\my-fake-news-app\backend\models\preprocessing_methods")
import pytest
from flask import Flask
from app import app  # Import your Flask app
from unittest.mock import patch, MagicMock
import json
import numpy as np
import time

# basic test
def test_homepage():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'<form' in response.data 


def test_predict_get():
    client = app.test_client()
    response = client.get('/predict')
    assert response.status_code == 200
    data = json.loads(response.data) 
    assert data["message"] == "Use POST with JSON data to get a prediction."

def test_predict_post_empty():
    client = app.test_client()
    response = client.post('/predict', data={'text': '', 'model': 'logistic'})
    assert response.status_code == 200
    assert b'Please enter text' in response.data

@pytest.mark.parametrize("model_name", ["lr", "nb", "svm", "rf", "dt", "lstm", "cnn-lstm", "bert", "bert-lstm"])
def test_all_models_with_valid_input(model_name):
    client = app.test_client()
    response = client.post('/predict', data={
        'text': 'Sample news text for testing',
        'model': model_name
    })
    assert response.status_code == 200
    assert b'Prediction:' in response.data


def test_prediction_consistency():
    client = app.test_client()
    text = "The sky is blue"
    responses = set()
    for _ in range(5):
        response = client.post('/predict', data={'text': text, 'model': 'lr'})
        responses.add(response.data)
    assert len(responses) == 1 

def test_html_tags_in_input():
    client = app.test_client()
    malicious_input = '<script>alert("XSS")</script>'
    response = client.post('/predict', data={
        'text': malicious_input,
        'model': 'lr'
    })
    assert response.status_code == 200
    assert b'<script>alert("XSS")</script>' not in response.data
    
    decoded_response = response.data.decode('utf-8')
    assert '&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;' in decoded_response or \
           malicious_input not in decoded_response
 
def test_very_long_input():
    client = app.test_client()
    long_text = "a" * 10000 
    response = client.post('/predict', data={
        'text': long_text,
        'model': 'lr'
    })
    assert response.status_code == 200

def test_lstm_with_rare_words():
    client = app.test_client()
    response = client.post('/predict', data={
        'text': "Supercalifragilisticexpialidocious quantum fluctuation",
        'model': 'lstm'
    })
    assert response.status_code == 200

def test_bert_with_special_characters():
    client = app.test_client()
    response = client.post('/predict', data={
        'text': "#@$%&)(*&^%$#@!)+_=-?><.,:;][]",
        'model': 'bert'
    })
    assert response.status_code == 200

def test_prediction_response_time():
    client = app.test_client()
    start_time = time.time()
    response = client.post('/predict', data={
        'text': "Sample news text",
        'model': 'lr'
    })
    elapsed = time.time() - start_time
    assert elapsed < 1.0  
    assert response.status_code == 200

#@pytest.mark.skip(reason="For load testing only")
def test_concurrent_requests():
    import threading
    client = app.test_client()
    results = []
    
    def make_request():
        response = client.post('/predict', data={
            'text': "Concurrent test",
            'model': 'lr'
        })
        results.append(response.status_code)
    
    threads = [threading.Thread(target=make_request) for _ in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    
    assert all(code == 200 for code in results)


def test_cors_headers():
    client = app.test_client()
    response = client.post('/predict', 
        data={'text': 'test', 'model': 'lr'},
        headers={'Origin': 'http://localhost:5173'}
    )
    assert response.headers.get('Access-Control-Allow-Origin') == 'http://localhost:5173'

def test_sql_injection_attempt():
    client = app.test_client()
    response = client.post('/predict', data={
        'text': "' OR 1=1 --",
        'model': 'lr'
    })
    assert response.status_code == 200
    assert b'SQL' not in response.data  

def test_form_rendering():
    client = app.test_client()
    response = client.get('/')
    assert b'<form' in response.data
    assert b'<textarea' in response.data
    assert b'<select' in response.data

def test_model_selection_persistence():
    client = app.test_client()
    response = client.post('/predict', data={
        'text': 'test',
        'model': 'bert'
    })
    decoded = response.data.decode('utf-8')
    
    assert 'value="bert"' in decoded
    assert 'value="bert"' in decoded and 'selected' in decoded.split('value="bert"')[1].split('>')[0]


@patch('app.LR.predict', side_effect=Exception("Model failed"))
def test_model_failure_handling(mock_predict):
    client = app.test_client()
    response = client.post('/predict', data={
        'text': 'test',
        'model': 'lr'
    })
    assert b'Error occurred' in response.data

def test_invalid_model_handling():
    client = app.test_client()
    response = client.post('/predict', data={
        'text': 'test',
        'model': 'invalid_model'
    })
    assert b'Invalid model selected' in response.data



@patch('app.tfidf_vectorizer.transform')
@patch('app.LR', new_callable=MagicMock)
def test_predict_post_logistic(mock_model, mock_vectorizer):
    mock_vectorizer.return_value = "mock_vectorized_text"
    mock_model.predict.return_value = [0]  
    mock_model.predict_proba.return_value = [[0.8, 0.2]]  

    client = app.test_client()
    response = client.post('/predict', data={'text': 'Fake news example', 'model': 'lr'})
    assert response.status_code == 200
    assert b'Prediction: Fake' in response.data
    assert b'Confidence: 80.00%' in response.data


@patch('app.tfidf_vectorizer.transform')
@patch('app.NB', new_callable=MagicMock)
def test_predict_post_naive_bayes(mock_model, mock_transform):
    mock_transform.return_value = "mock_vectorized_text"
    mock_model.predict.return_value = [1]  
    mock_model.predict_proba.return_value = [[0.3, 0.7]]  
    
    client = app.test_client()
    response = client.post('/predict', data={'text': 'Real news example', 'model': 'nb'})
    
    assert response.status_code == 200
    assert b'Prediction: Real' in response.data
    assert b'Confidence: 70.00%' in response.data

@patch('app.tfidf_vectorizer.transform')
@patch('app.SVM', new_callable=MagicMock)
def test_predict_post_svm(mock_model, mock_transform):
    mock_transform.return_value = "mock_vectorized_text"
    mock_model.predict.return_value = [0]  # Fake prediction
    mock_model.predict_proba.return_value = [[0.9, 0.1]]  # 90% confidence
    
    client = app.test_client()
    response = client.post('/predict', data={'text': 'Fake news example', 'model': 'svm'})
    
    assert response.status_code == 200
    assert b'Prediction: Fake' in response.data
    assert b'Confidence: 90.00%' in response.data

@patch('app.tfidf_vectorizer.transform')
@patch('app.RF', new_callable=MagicMock)
def test_predict_post_random_forest(mock_model, mock_transform):
    mock_transform.return_value = "mock_vectorized_text"
    mock_model.predict.return_value = [1]  # Real prediction
    mock_model.predict_proba.return_value = [[0.45, 0.55]]  # 55% confidence
    
    client = app.test_client()
    response = client.post('/predict', data={'text': 'Borderline news', 'model': 'rf'})
    
    assert response.status_code == 200
    assert b'Prediction: Real' in response.data
    assert b'Confidence: 55.00%' in response.data

@patch('app.tfidf_vectorizer.transform')
@patch('app.DT', new_callable=MagicMock)
def test_predict_post_decision_tree(mock_model, mock_transform):
    mock_transform.return_value = "mock_vectorized_text"
    mock_model.predict.return_value = [0]  # Fake prediction
    mock_model.predict_proba.return_value = [[0.95, 0.05]]  # 95% confidence
    
    client = app.test_client()
    response = client.post('/predict', data={'text': 'Fake news example', 'model': 'dt'})
    
    assert response.status_code == 200
    assert b'Prediction: Fake' in response.data
    assert b'Confidence: 95.00%' in response.data


@patch('app.lstm_tokenizer.texts_to_sequences')
@patch('app.LSTM.predict')
def test_predict_post_lstm(mock_predict, mock_tokenize):
    mock_tokenize.return_value = [[1, 2, 3]]  # Mock tokenized sequence
    mock_predict.return_value = np.array([[0.4]])  # 40% probability (Fake)
    
    client = app.test_client()
    response = client.post('/predict', data={'text': 'Fake news example', 'model': 'lstm'})
    
    assert response.status_code == 200
    assert b'Prediction: Fake' in response.data
    assert b'Confidence: 60.00%' in response.data  

@patch('app.lstm_tokenizer.texts_to_sequences')
@patch('app.CNN_LSTM.predict')
def test_predict_post_cnn_lstm(mock_predict, mock_tokenize):
    mock_tokenize.return_value = [[1, 2, 3]]
    mock_predict.return_value = np.array([[0.7]]) 
    
    client = app.test_client()
    response = client.post('/predict', data={'text': 'Real news example', 'model': 'cnn-lstm'})
    
    assert response.status_code == 200
    assert b'Prediction: Real' in response.data
    assert b'Confidence: 70.00%' in response.data


@patch('app.bert_predict')
def test_predict_post_bert(mock_bert_predict):

    mock_bert_predict.return_value = 0.15  # Fake news, 85% confidence

    client = app.test_client()
    response = client.post('/predict', json={'text': 'This is fake', 'model': 'bert'})
    
    print(response.data.decode())

    assert response.status_code == 200
    assert b'Prediction: Fake' in response.data
    assert b'Confidence: 85.00%' in response.data

@patch('app.bert_predict')
def test_predict_post_bert_lstm(mock_bert_predict):
    mock_bert_predict.return_value = 0.65  # 65% probability (Real)
    
    client = app.test_client()
    response = client.post('/predict', data={'text': 'Real news example', 'model': 'bert-lstm'})
    
    assert response.status_code == 200
    assert b'Prediction: Real' in response.data
    assert b'Confidence: 65.00%' in response.data


def test_predict_post_invalid_model():
    client = app.test_client()
    response = client.post('/predict', data={'text': 'Hello', 'model': 'random_model'})
    assert response.status_code == 200
    assert b'Invalid model selected' in response.data


@patch('app.bert_predict', side_effect=Exception("BERT model error"))
def test_bert_prediction_exception(mock_bert):
    client = app.test_client()
    response = client.post('/predict', data={'text': 'test text', 'model': 'bert'})

    assert response.status_code == 200
    assert (b'BERT Error: BERT model error' in response.data or 
            b'Error occurred: BERT model error' in response.data or
            b'Error occurred during prediction' in response.data)
    #assert b'BERT Error: BERT model error' in response.data

if __name__ == "__main__":
    pytest.main()
