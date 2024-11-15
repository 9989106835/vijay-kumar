import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Define the path to the CSV file
file_path = 'data/spam_data.csv'

# Check if the file exists
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("Data loaded successfully!")
    print(data.head())  # Display first few rows
else:
    print("File does not exist.")
    exit()

# Check if the DataFrame is empty
if data.empty:
    print("The DataFrame is empty. Check the contents of the CSV file.")
else:
    print("DataFrame columns:", data.columns)

    if 'text' in data.columns and 'label' in data.columns:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

        # Vectorize the text data
        vectorizer = CountVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        # Train the model
        model = MultinomialNB()
        model.fit(X_train_vectorized, y_train)

        # Evaluate the model
        accuracy = model.score(X_test_vectorized, y_test)
        print(f'Model accuracy: {accuracy:.2f}')
    else:
        print("Expected columns 'text' and 'label' not found.")
