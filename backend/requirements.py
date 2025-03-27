# pip install flask transformers scikit-learn joblib tensorflow
# python.exe -m pip install pandas scikit-learn joblib nltk transformers
# python.exe -m pip install tf-keras
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
CNN_LSTM = load_model("models/cnn_lstm_fixed.keras")


# Verify your CNN_LSTM model loaded correctly
# print(CNN_LSTM.summary())  # Should show model architecture
# test_pred = CNN_LSTM.predict(np.zeros((1, 200)))  # Should return valid prediction

# Add this check right after loading your model
# print("Verifying CNN-LSTM model...")
#print(CNN_LSTM.summary())

# Test with dummy data
# test_input = np.zeros((1, 200))  # Match your max_len
# test_pred = CNN_LSTM.predict(test_input)
# print(f"Test prediction shape: {test_pred.shape}, value: {test_pred[0][0]}")
sample_text="Head of a conservative Republican faction in the U.S. Congress urged budget restraint in 2019"

with open('models/preprocessing_methods/LSTMs-tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

sequence = tokenizer.texts_to_sequences([sample_text])
#padded_sample = pad_sequences(sequences, maxlen=max_len)
padded_sample = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')

cnn_lstm_new_pred = CNN_LSTM.predict(padded_sample)
print(f"Raw model output: {cnn_lstm_new_pred}")
print("CNN-LSTM Max Layer Prediction:", "Fake" if cnn_lstm_new_pred[0][0] >= 0.5 else "Real")

if cnn_lstm_new_pred[0][0] >= 0.5:
    print("Max CNN-LSTM Prediction: REAL NEWS (confidence: {:.2f}%)".format(cnn_lstm_new_pred[0][0] * 100))
else:
    print("Max CNN-LSTM Prediction: FAKE NEWS (confidence: {:.2f}%)".format((1 - cnn_lstm_new_pred[0][0]) * 100))

prediction = cnn_lstm_new_pred[0][0]
print(f"Final prediction value: {prediction}")  # Debug
                
            # if not (0 <= prediction <= 1):
            #     raise ValueError(f"Prediction {prediction} out of [0,1] range")         
result = "Fake" if prediction < 0.5 else "Real"
confidence = f"{max(prediction, 1-prediction)*100:.2f}%"

print(f"Result : {result}")
print(f"Confidence : {confidence}")
            