# pip install flask transformers scikit-learn joblib tensorflow
# python.exe -m pip install pandas scikit-learn joblib nltk transformers
# python.exe -m pip install tf-keras
import numpy as np
from tensorflow.keras.models import load_model
CNN_LSTM = load_model("models/cnn_lstm_test.keras")

# Verify your CNN_LSTM model loaded correctly
# print(CNN_LSTM.summary())  # Should show model architecture
# test_pred = CNN_LSTM.predict(np.zeros((1, 200)))  # Should return valid prediction

# Add this check right after loading your model
# print("Verifying CNN-LSTM model...")
# print(CNN_LSTM.summary())

# Test with dummy data
test_input = np.zeros((1, 200))  # Match your max_len
test_pred = CNN_LSTM.predict(test_input)
print(f"Test prediction shape: {test_pred.shape}, value: {test_pred[0][0]}")