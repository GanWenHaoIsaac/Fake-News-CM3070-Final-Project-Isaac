import sys
import os
sys.path.insert(0, "Users\isaac\Desktop\CM3070-FakeNews-Vite-App\my-fake-news-app\backend\models")
sys.path.insert(0, "Users\isaac\Desktop\CM3070-FakeNews-Vite-App\my-fake-news-app\backend\models\preprocessing_methods")
import pytest
from flask import Flask
from app import app
from unittest.mock import patch, MagicMock
import json
import numpy as np
import time

def test_homepage():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["message"] == "Use POST with JSON data to get a prediction."


def test_predict_get():
    client = app.test_client()
    response = client.get('/predict')
    assert response.status_code == 200
    data = json.loads(response.data) 
    assert data["message"] == "Use POST with JSON data to get a prediction."

def test_predict_post_empty():
    client = app.test_client()
    response = client.post('/predict', json={'text': '', 'model': 'lr'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "error" in data
    assert data["error"] == "No text provided."

@pytest.mark.parametrize("model_name", ["lr", "nb", "svm", "rf", "dt", "lstm", "cnn-lstm", "bert", "bert-lstm"])
def test_all_models_with_valid_input(model_name):
    client = app.test_client()
    response = client.post('/predict', json={
        'text': 'Sample news text for testing',
        'model': model_name
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "prediction" in data
    assert "confidence" in data


def test_prediction_consistency():
    client = app.test_client()
    text = "The sky is blue"
    responses = set()
    for _ in range(5):
        response = client.post('/predict', json={'text': text, 'model': 'lr'})
        responses.add(response.data)
    assert len(responses) == 1 

def test_html_tags_in_input():
    client = app.test_client()
    malicious_input = '<script>alert("XSS")</script>'
    response = client.post('/predict', json={
        'text': malicious_input,
        'model': 'lr'
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "<script>" not in data["prediction"]
 
def test_very_long_input():
    client = app.test_client()
    long_text = "a" * 10000 
    response = client.post('/predict', json={
        'text': long_text,
        'model': 'lr'
    })
    assert response.status_code == 200

def test_lstm_with_rare_words():
    client = app.test_client()
    response = client.post('/predict', json={
        'text': "Consanguineous agathokakological Psychotomimetic",
        'model': 'lstm'
    })
    assert response.status_code == 200

def test_bert_with_special_characters():
    client = app.test_client()
    response = client.post('/predict', json={
        'text': "#@$%&)(*&^%$#@!)+_=-?><.,:;][]",
        'model': 'bert'
    })
    assert response.status_code == 200

def test_prediction_response_time():
    client = app.test_client()
    start_time = time.time()
    response = client.post('/predict', json={
        'text': "Sample news text",
        'model': 'lr'
    })
    elapsed = time.time() - start_time
    assert elapsed < 1.0  
    assert response.status_code == 200

def test_concurrent_requests():
    import threading
    client = app.test_client()
    results = []
    
    def make_request():
        response = client.post('/predict', json={
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
        json={'text': 'test', 'model': 'lr'},
        headers={'Origin': 'http://localhost:5173'}
    )
    assert response.headers.get('Access-Control-Allow-Origin') == 'http://localhost:5173'


def test_form_rendering():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'"message":"Use POST with JSON data to get a prediction."' in response.data

def test_model_selection_persistence():
    client = app.test_client()
    response = client.post('/predict', json={
        'text': 'test',
        'model': 'bert'
    })
    data = response.get_json() 
    
    assert response.status_code == 200
    assert data["model"] == "bert"

def test_invalid_model_handling():
    client = app.test_client()
    response = client.post('/predict', json={
        'text': 'test',
        'model': 'invalid_model'
    })
    assert b'Invalid model selected' in response.data


@patch('app.joblib.load')
@patch('app.load_model')
def test_predict_post_logistic(mock_load, mock_keras_load):
    client = app.test_client()
    response = client.post('/predict', json={'text': 'Head of a conservative Republican faction in the U.S. Congress urged budget restraint in 2019', 'model': 'lr'})
    assert response.status_code == 200
    response_json = json.loads(response.data.decode())
    assert response_json['prediction'] == 'Real'
    assert response_json['model'] == 'lr'


@patch('app.joblib.load')
@patch('app.load_model')
def test_predict_post_naive_bayes(mock_load, mock_keras_load):
    mock_load.return_value = "mock_naive_bayes_model"
    mock_keras_load.return_value = "mock_keras_model"  
    
    client = app.test_client()
    response = client.post('/predict', json={'text': 'Head of a conservative Republican faction in the U.S. Congress urged budget restraint in 2019', 'model': 'nb'})
    
    assert response.status_code == 200
    response_json = json.loads(response.data.decode())

    # Check the content of the response
    assert response_json['prediction'] == 'Real'
    assert response_json['model'] == 'nb'


@patch('app.joblib.load')
@patch('app.load_model')
def test_predict_post_svm(mock_load, mock_keras_load):
    client = app.test_client()
    response = client.post('/predict', json={'text': 'Fake news example', 'model': 'svm'})   
    assert response.status_code == 200
    response_json = json.loads(response.data.decode())
    assert response_json['prediction'] == 'Fake'
    assert response_json['model'] == 'svm'

@patch('app.joblib.load')
@patch('app.load_model')
def test_predict_post_random_forest(mock_load, mock_keras_load):

    client = app.test_client()
    response = client.post('/predict', json={'text': 'Fake News', 'model': 'rf'})
    
    assert response.status_code == 200
    response_json = json.loads(response.data.decode())

    assert response_json['prediction'] == 'Fake'
    assert response_json['model'] == 'rf'


@patch('app.joblib.load')
@patch('app.load_model')
def test_predict_post_decision_tree(mock_load, mock_keras_load):
    
    client = app.test_client()
    response = client.post('/predict', json={'text': 'Fake news example', 'model': 'dt'})
    
    assert response.status_code == 200
    response_json = json.loads(response.data.decode())

    assert response_json['prediction'] == 'Fake'
    assert response_json['model'] == 'dt'


@pytest.mark.parametrize("model_name", ["lstm", "cnn-lstm"])
@patch('app.load_model')
@patch('app.lstm_tokenizer.texts_to_sequences')
def test_lstm_models(mock_texts_to_sequences, mock_load_model, model_name):
    mock_model = mock_load_model.return_value
    mock_texts_to_sequences.return_value = [[1, 2, 3, 4, 5]]
    
    text = "This is a sample fake news article to test the model."
    input_data = mock_texts_to_sequences([text])
    
    mock_model.predict.return_value = np.array([[0.7]]) 
    probability = mock_model.predict(input_data)[0][0]
    
    assert 0 <= probability <= 1, f"{model_name} produced an invalid probability: {probability}"
    
    result = "Real" if probability >= 0.5 else "Fake"
    confidence = f"{(1 - probability) * 100:.2f}%" if result == "Fake" else f"{probability * 100:.2f}%"
    
    assert isinstance(result, str), "Result should be a string."
    assert result in ["Real", "Fake"], "Result should be either 'Real' or 'Fake'."
    assert isinstance(confidence, str) and confidence.endswith("%"), "Confidence should be a percentage string."


@patch('app.bert_predict')
def test_predict_post_bert(mock_bert_predict):

    mock_bert_predict.return_value = 0.15  

    client = app.test_client()
    response = client.post('/predict', json={'text': 'This is fake', 'model': 'bert'})
    
    print(response.data.decode())

    assert response.status_code == 200
    response_json = json.loads(response.data.decode())

    assert response_json['prediction'] == 'Fake'
    assert response_json['confidence'] == '85.00%'
    assert response_json['model'] == 'bert'

@patch('app.bert_predict')
def test_predict_post_bert_lstm(mock_bert_predict):
    mock_bert_predict.return_value = 0.65 
    
    client = app.test_client()
    response = client.post('/predict', json={'text': 'Real news example', 'model': 'bert-lstm'})
    
    assert response.status_code == 200
    response_json = json.loads(response.data.decode())

    # Check the content of the response
    assert response_json['prediction'] == 'Real'
    assert response_json['confidence'] == '65.00%'
    assert response_json['model'] == 'bert-lstm'


@patch('app.bert_predict', side_effect=Exception("BERT model error"))
def test_bert_prediction_exception(mock_bert):
    client = app.test_client()
    response = client.post('/predict', json={'text': 'test text', 'model': 'bert'})
    assert response.status_code == 500
    response_json = json.loads(response.data.decode())
    assert response_json['error'] == 'An error occurred during prediction.'


def test_predict_post_invalid_model():
    client = app.test_client()
    response = client.post('/predict', json={'text': 'Hello', 'model': 'random_model'})
    assert response.status_code == 400
    assert b'Invalid model selected' in response.data

if __name__ == "__main__":
    pytest.main()
