import os
import requests
from zipfile import ZipFile
import pandas as pd
from keras.initializers.initializers import Constant
from keras.utils import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Dense, LSTM
from keras.optimizers import Adam

from keras.callbacks import EarlyStopping
import numpy as np
import pickle

nltk.download('punkt')
nltk.download('wordnet')


# Function to download and extract GloVe embeddings
def download_glove():
    if not os.path.exists('glove.6B.100d.txt'):
        glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
        response = requests.get(glove_url)
        with open('../010_prompt_002/glove.6B.zip', 'wb') as f:
            f.write(response.content)
        with ZipFile('../010_prompt_002/glove.6B.zip', 'r') as zip_ref:
            zip_ref.extractall()


# Call the function to download and extract GloVe embeddings
download_glove()

# Load the data
data = pd.read_pickle('../007_alt_model/data.pkl')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


# Define a function to lemmatize words
def lemmatize_words(text):
    words = word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(word) for word in words)


# Apply the function to the 'spells' column
data['spells'] = data['spells'].apply(lemmatize_words)

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=1908, oov_token="<OOV>")
tokenizer.fit_on_texts(data['spells'])

# Save tokenizer for prediction
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert text into sequence of tokens
sequence = tokenizer.texts_to_sequences(data['spells'])
padded_sequence = pad_sequences(sequence, maxlen=150)

# Load the GloVe embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

# Prepare the embedding matrix
num_tokens = len(tokenizer.word_index) + 2
embedding_dim = 100
hits = 0
misses = 0
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1

# Define the model
model = Sequential()
model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(embedding_matrix), trainable=False))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))  # LSTM layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

# Compile the model
model.compile(loss='mean_absolute_error', optimizer=Adam(), metrics=['mae'])

# Train the model
model.fit(padded_sequence, data['price_in_keys'], validation_split=0.2, epochs=20,
          callbacks=[EarlyStopping(patience=3)])

# Save the model for prediction
model.save("trained_model")
