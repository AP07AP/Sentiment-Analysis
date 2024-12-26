import streamlit as st
import numpy as np
import gensim
from gensim.models import Word2Vec
import nltk
import joblib  # Import joblib to load the saved model
from sklearn.preprocessing import normalize
# Load your pre-trained Word2Vec model
model = Word2Vec.load("word2vec_model")  # Path to your saved Word2Vec model

# Tokenizer function
def tokenize_text(text):
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    return tokens

# Function to convert a review to a vector using Word2Vec
def review_to_vector(review, model):
    tokens = tokenize_text(review)  # Tokenize the review text
    # st.write("Tokens:", tokens)  # Display tokens in the GUI
    
    vector = np.zeros(100)  # Initialize an empty vector of size 100 (word vector size)
    count = 0
    
    # Create a dictionary to store word vectors for each token
    word_vectors = {}

    # Loop through each token (word) in the review
    for word in tokens:
        if word in model.wv:  # Check if the word exists in the Word2Vec vocabulary
            word_vectors[word] = model.wv[word]  # Save the word vector in the dictionary
            vector += model.wv[word]  # Add the word vector to the review vector
            count += 1  # Increment the count for valid words found in the Word2Vec model
            
    # If valid words were found, average the word vectors
    if count > 0:
        vector /= count  # Normalize the vector by averaging the word vectors
    
    # Display the word vectors for each token
    # st.write("Word Vectors for each token:")
    # for word, vec in word_vectors.items():
    #     st.write(f"Word: {word}, Vector: {vec[:10]}...")  # Display the first 10 elements of the vector for clarity
    # Normalize the vector (optional step)
    vector = normalize([vector])[0]
    
    # st.write("Aggregated Vector for Review: ", vector)  
    # Return the vector representing the review
    return vector

# Load the logistic regression model using joblib
logistic_model = joblib.load("logistic_model.pkl")  # Correct way to load the model

# Streamlit GUI
st.title("Sentiment Analysis")

# Text input for user to enter a review
input_text = st.text_area("Enter your review here:")

# Add a submit button
if st.button("Submit"):
    if input_text:
        # Convert the input text to a Word2Vec vector
        input_vector = review_to_vector(input_text, model)
        
        # Predict the sentiment using the trained Logistic Regression model
        sentiment_pred = logistic_model.predict([input_vector])

        # Display the prediction
        if sentiment_pred == 'positive':
            st.write("The sentiment is **Positive**.")
        else:
            st.write("The sentiment is **Negative**.")
