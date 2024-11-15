from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

# Load your dataset and model
data = pd.read_csv('spam_data.csv')
X = data['text']
y = data['label']

# Train the model
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
model = MultinomialNB()
model.fit(X_vectorized, y)

# Save the model and vectorizer for later use
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('spam_model.pkl')

    # Vectorize the input message
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)

    # Return the prediction
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
