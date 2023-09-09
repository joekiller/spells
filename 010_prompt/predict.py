import pickle
from keras.models import load_model
from keras.utils import pad_sequences

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load model
model = load_model("trained_model")


def predict_price(spell):
    # Now you can use the tokenizer and model for prediction
    sequence = tokenizer.texts_to_sequences([spell])
    padded_sequence = pad_sequences(sequence, maxlen=150)
    prediction = model.predict(padded_sequence)[0]
    return prediction


# Prediction loop
while True:
    text = input("Enter a spell description: ")
    if text.lower() == "quit":
        break
    else:
        print("Predicted price_in_keys: ", predict_price(text))
