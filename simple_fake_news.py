import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("C:/Users/supri/Downloads/TASK-2/Fake_News_Detection-master/train.csv")


# Keep only required columns
text_col = data.columns[0]
label_col = data.columns[-1]

X = data[text_col]
y = data[label_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
predictions = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Fake News Detection Model Accuracy:", accuracy)
# ---- Custom Prediction ----
while True:
    user_input = input("\nEnter a news statement (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    user_vec = vectorizer.transform([user_input])
    result = model.predict(user_vec)

    if result[0] == 1:
        print("Prediction: REAL news")
    else:
        print("Prediction: FAKE news")
