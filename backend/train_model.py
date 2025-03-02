import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
import string

# Load fake and real news datasets
fake_news = pd.read_csv('data/Fake.csv')
real_news = pd.read_csv('data/True.csv')

fake_news.head()
real_news.head()

# Assign labels correctly
fake_news['label'] = 0  # Fake news = 0
real_news['label'] = 1  # Real news = 1
fake_news.shape, real_news.shape

#data = pd.concat([fake_news, real_news], ignore_index=True)
fake_manual_testing = fake_news.tail(10)
for i in range(23480, 23470, -1):
    fake_news.drop([i], axis = 0, inplace = True)

real_manual_testing = real_news.tail(10)
for i in range(21416, 21406, -1):
    real_news.drop([i], axis = 0, inplace = True)

fake_manual_testing['label'] = 0
real_manual_testing['label'] = 1

merge_news = pd.concat([fake_news, real_news], axis = 0)
merge_news.head(10)

data = merge_news.drop(['title', 'subject', 'date'], axis = 1)



# Balance the dataset by undersampling the majority class
# min_samples = min(len(fake_news), len(real_news))
# fake_news_balanced = fake_news.sample(n=min_samples, random_state=42)
# real_news_balanced = real_news.sample(n=min_samples, random_state=42)

# # Combine balanced datasets
# data_balanced = pd.concat([fake_news_balanced, real_news_balanced], ignore_index=True)

# # Shuffle the balanced dataset
# data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# # Verify class distribution
# print(data_balanced['label'].value_counts())

# Shuffle dataset
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)
# print(data['label'].value_counts())

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# def preprocess_text(text):
#     text = re.sub(r'\W', ' ', text)  # Remove special characters
#     text = re.sub(r'\d+', ' ', text)  # Remove numbers
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
#     text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatize and remove stopwords
#     return text

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?:\/\/\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(wordopt)

x = data['text']
y = data['label']

# Apply preprocessing
#data_balanced.dropna(inplace=True)  # Remove missing values
#data['text'] = data['text'].apply(preprocess_text)

# # Split dataset
# X = data['text']
# y = data['label']
# Apply preprocessing to the balanced dataset
#data_balanced['text'] = data_balanced['text'].apply(preprocess_text)

# Split the balanced dataset
# X = data_balanced['text']
# y = data_balanced['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Convert text into TF-IDF vectors
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(x_train)
X_test_tfidf = tfidf.transform(x_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate model
lr_pred = model.predict(X_test_tfidf)
model.score(X_test_tfidf, y_test)
print("LR model:")
print(classification_report(y_test, lr_pred))
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.4f}")

# Save the model and vectorizer
# import os
# os.makedirs("models", exist_ok=True)

DT = DecisionTreeClassifier()
DT.fit(X_train_tfidf, y_train)

dt_pred = DT.predict(X_test_tfidf)
DT.score(X_test_tfidf, y_test)
print("DT: ", classification_report(y_test, dt_pred))


def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = tfidf.transform(new_x_test)
    
    lr_pred = model.predict(new_xv_test)
    dt_pred = DT.predict(new_xv_test)

    # return print("\n\nLR Prediction: {} \nDT Prediction:".format(
    #     output_label(lr_pred[0]), 
    #     output_label(dt_pred[0])
    # ))
    print(f"\n\nLR Prediction: {output_label(lr_pred[0])} \nDT Prediction: {output_label(dt_pred[0])}")

news = str(input())
manual_testing(news)


# print("Saving model to models/logistic_regression.pkl...")
# joblib.dump(model, "models/logistic_regression.pkl")
# print("Model saved successfully.")

# print("Saving vectorizer to models/tfidf_vectorizer.pkl...")
# joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
# print("Vectorizer saved successfully.")

# print("Logistic Regression model and TF-IDF vectorizer saved successfully.")

# test_samples = [
#     "The government is secretly controlled by aliens, and they are planning to reveal themselves next week.",  # Fake
#     "NASA's Perseverance rover successfully landed on Mars in February 2021.",  # Real
#     "A new study shows that drinking coffee can extend your life by 10 years.",  # Fake
#     "The United Nations has called for global action to combat climate change.",  # Real
# ]

# for sample in test_samples:
#     processed_sample = preprocess_text(sample)
#     sample_vector = tfidf.transform([processed_sample])
#     prediction = model.predict(sample_vector)
#     print(f"Text: {sample}\nPrediction: {'Fake' if prediction[0] == 1 else 'Real'}\n")

# import pandas as pd
# import joblib
# import re
# import nltk
# from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# # Load fake and real news datasets
# fake_news = pd.read_csv('data/Fake.csv')
# real_news = pd.read_csv('data/True.csv')
 
# # Assign labels
# fake_news['label'] = 0  # Fake news
# real_news['label'] = 1  # Real news
# data = pd.concat([fake_news, real_news], ignore_index=True)

# # Shuffle dataset
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# # Download NLTK stopwords
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# # Text preprocessing function
# def preprocess_text(text):
#     text = re.sub(r'\W', ' ', text)  # Remove special characters
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
#     text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
#     return text

# # Apply preprocessing
# data.dropna(inplace=True)  # Remove missing values
# data['text'] = data['text'].apply(preprocess_text)

# # Split dataset
# X = data['text']
# y = data['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Convert text into TF-IDF vectors
# tfidf = TfidfVectorizer(max_features=5000)
# X_train_tfidf = tfidf.fit_transform(X_train)
# X_test_tfidf = tfidf.transform(X_test)

# # Train Logistic Regression model
# model = LogisticRegression()
# model.fit(X_train_tfidf, y_train)

# # Evaluate model
# y_pred = model.predict(X_test_tfidf)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.4f}")

# import os

# # Ensure the 'models' directory exists
# os.makedirs("models", exist_ok=True)


# print("Saving model to models/logistic_regression.pkl...")
# joblib.dump(model, "models/logistic_regression.pkl")
# print("Model saved successfully.")

# print("Saving vectorizer to models/tfidf_vectorizer.pkl...")
# joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
# print("Vectorizer saved successfully.")


# print("Logistic Regression model and TF-IDF vectorizer saved successfully.")
