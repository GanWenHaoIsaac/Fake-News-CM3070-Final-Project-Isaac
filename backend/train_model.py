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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import string

# Load fake and real news datasets
fake_news = pd.read_csv('data/fake_cleaned.csv')
real_news = pd.read_csv('data/true_cleaned.csv')

fake_news.head()
real_news.head()

# Assign labels correctly
fake_news['label'] = 0  # Fake news = 0
real_news['label'] = 1  # Real news = 1
fake_news.shape, real_news.shape

#data = pd.concat([fake_news, real_news], ignore_index=True)
# fake_manual_testing = fake_news.tail(10)
# real_manual_testing = real_news.tail(10)
# for i in range(21416, 21406, -1):
#     real_news.drop([i], axis = 0, inplace = True)
# fake_news.reset_index(drop=True, inplace=True)
# real_news.reset_index(drop=True, inplace=True)


# fake_manual_testing['label'] = 0
# real_manual_testing['label'] = 1

merge_news = pd.concat([fake_news, real_news], axis = 0)
merge_news.head(10)

data = merge_news.drop(['title', 'subject', 'date'], axis = 1)

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


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Convert text into TF-IDF vectors
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(x_train)
X_test_tfidf = tfidf.transform(x_test)

# Train Logistic Regression model
LR = LogisticRegression()
LR.fit(X_train_tfidf, y_train)

# Evaluate model
lr_pred = LR.predict(X_test_tfidf)
LR.score(X_test_tfidf, y_test)
print("LR model:")
print(classification_report(y_test, lr_pred))
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.4f}")

# Save the model and vectorizer
DT = DecisionTreeClassifier()
DT.fit(X_train_tfidf, y_train)

dt_pred = DT.predict(X_test_tfidf)
DT.score(X_test_tfidf, y_test)
print("DT: ", classification_report(y_test, dt_pred))



# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# # Evaluate Random Forest model
# rf_pred = rf_model.predict(X_test_tfidf)
# print("Random Forest Classification Report:")
# print(classification_report(y_test, rf_pred))

# Train SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# # Evaluate SVM model
# svm_pred = svm_model.predict(X_test_tfidf)
# print("SVM Classification Report:")
# print(classification_report(y_test, svm_pred))


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
    
    lr_pred = LR.predict(new_xv_test)
    dt_pred = DT.predict(new_xv_test)
    # rf_pred = rf_model.predict(new_xv_test)
    # svm_pred = svm_model.predict(new_xv_test)

    print(f"\n\nLR Prediction: {output_label(lr_pred[0])}")
    print(f"DT Prediction: {output_label(dt_pred[0])}")
    print(f"RF Prediction: {output_label(rf_pred[0])}")
    print(f"SVM Prediction: {output_label(svm_pred[0])}")


# news = str(input("Enter news: "))
# manual_testing(news)

# Save the models and vectorizer
import os
os.makedirs("models", exist_ok=True)

print("Saving model to models/logistic_regression.pkl...")
joblib.dump(LR, "models/logistic_regression.pkl")
print("LR model saved successfully.")

print("Saving vectorizer to models/decision_tree.pkl...")
joblib.dump(DT, "models/decision_tree.pkl")
print("DT model saved successfully.")

print("Saving model to models/logistic_regression.pkl...")
joblib.dump(rf_model, "models/random_forest.pkl")
print("LR model saved successfully.")

print("Saving vectorizer to models/decision_tree.pkl...")
joblib.dump(svm_model, "models/svm_model.pkl")
print("DT model saved successfully.")

print("Saving TF-IDF vectorizer to models/tfidf_vectorizer.pkl...")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
print("TF-IDF vectorizer saved successfully.")

