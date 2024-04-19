import pandas as pd
import argparse
import re
import numpy as np
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from keras.models import load_model
from collections import Counter

# Load the model
model = load_model('LSTM_FINAL.h5')

#Functions for data preprocessing
def text_preprocessing(text):
    # Normalize text to lowercase
    text = text.lower()
    # Remove anything that is not a UTF-8 character
    text = text.encode("utf-8", "ignore").decode("utf-8")
    # Use regular expressions to remove all characters that are not letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Split text into words (tokens)
    words = text.split()
    # Load a set of stopwords
    stop_words = set(stopwords.words('english'))
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

def pad_features(encoded_texts, sequence_length):
    features = np.zeros((len(encoded_texts), sequence_length), dtype=int)
    for i, row in enumerate(encoded_texts):
        features[i, -len(row):] = np.array(row)[:sequence_length]
    return features

def load_data(filename):
    df = pd.read_csv(filename)
    df['processed_text'] = df['Sentence'].apply(text_preprocessing)
    return df

def analyze_user_input(text, model, vocab_to_int, sequence_length):
    # Preprocess the text
    processed_text = text_preprocessing(text)
    
    # Encode the preprocessed text with the existing vocab_to_int mapping
    encoded_text = [vocab_to_int.get(word, 0) for word in processed_text]
    
    # Pad the encoded text
    padded_text = pad_features([encoded_text], sequence_length)
    
    # Predict the sentiment
    prediction = model.predict(np.array(padded_text))
    predicted_sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    
    # Output result
    return predicted_sentiment

def create_vocab_and_encode(df):
    all_words = [word for text in df['processed_text'] for word in text]
    counts = Counter(all_words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: i+1 for i, word in enumerate(vocab)}  # ensure indexing starts from 1
    df['encoded'] = df['processed_text'].apply(lambda x: [vocab_to_int[word] for word in x])
    vocab_size = len(vocab_to_int) 
    return df, vocab_to_int, vocab_size


def main(user_input):
    df = load_data('output_from_amazon_imdb_yelp.csv')
    df, vocab_to_int, vocab_size = create_vocab_and_encode(df)
    sequence_length = 50

    sentiment_result = analyze_user_input(user_input, model, vocab_to_int, sequence_length)
    if sentiment_result == 'Negative':
        print(f"Prompt to chatbot: The user seems to be feeling down based on their last message '{user_input}'. Provide some supportive responses or ask if they'd like to talk about what's bothering them?")
    else:
        print(f"Prompt to chatbot: The user appears to be feeling good based on their last message '{user_input}'. Keep the conversation light and positive.")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze sentiment of user input.')
    parser.add_argument('user_input', type=str, help='provide user input for prompt production')
    args = parser.parse_args()

    main(args.user_input)