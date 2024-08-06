import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

data1 = pd.read_csv(r'C:\Users\pavan\Documents\project\Movie Dataset\test_data.csv', low_memory=False)
data2 = pd.read_csv(r'C:\Users\pavan\Documents\project\Movie Dataset\test_data_solution.csv', low_memory=False)
data3 = pd.read_csv(r'C:\Users\pavan\Documents\project\Movie Dataset\train_data.csv', low_memory=False)

print("Data1 columns:", data1.columns)
print("Data2 columns:", data2.columns)
print("Data3 columns:", data3.columns)

data1.columns = ['identifier', 'headline', 'summary']
data2.columns = ['identifier', 'headline', 'category', 'summary']
data3.columns = ['identifier', 'headline', 'category', 'summary']

combined_data = pd.concat([data1, data2, data3], ignore_index=True)

print(combined_data.head())

summary_col = 'summary'
category_col = 'category'

if summary_col in combined_data.columns and category_col in combined_data.columns:
    combined_data[summary_col] = combined_data[summary_col].fillna('')
    combined_data = combined_data.dropna(subset=[category_col])

    label_encoder = LabelEncoder()
    combined_data['encoded_category'] = label_encoder.fit_transform(combined_data[category_col])

    train_set, test_set = train_test_split(combined_data, test_size=0.2, random_state=42)

    X_train_text = train_set[summary_col]
    y_train_labels = train_set['encoded_category']
    X_test_text = test_set[summary_col]
    y_test_labels = label_encoder.transform(test_set[category_col])

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf_features = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf_features = tfidf_vectorizer.transform(X_test_text)

    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf_features, y_train_labels)

    y_predictions = classifier.predict(X_test_tfidf_features)

    accuracy = accuracy_score(y_test_labels, y_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test_labels, y_predictions, target_names=label_encoder.classes_))
else:
    print(f"Columns '{summary_col}' and/or '{category_col}' not found in the combined dataset.")
