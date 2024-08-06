import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r'C:\Users\pavan\Documents\project\SCD dataset\SCD.csv')

print("First few rows of the dataset:")
print(df.head())
print("\nColumn names of the dataset:")
print(df.columns)

df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

df = df[['label', 'message']]

print("\nChecking for missing values:")
print(df.isnull().sum())
df.dropna(inplace=True)

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_tfidf = tfidf_vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

logistic_regression = LogisticRegression()
naive_bayes_classifier = MultinomialNB()
support_vector_machine = SVC()

logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)
print("\nLogistic Regression")
print(f'Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}')
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

naive_bayes_classifier.fit(X_train, y_train)
y_pred_nb = naive_bayes_classifier.predict(X_test)
print("\nNaive Bayes")
print(f'Accuracy: {accuracy_score(y_test, y_pred_nb):.2f}')
print(classification_report(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))

support_vector_machine.fit(X_train, y_train)
y_pred_svm = support_vector_machine.predict(X_test)
print("\nSupport Vector Machine")
print(f'Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}')
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))