import pickle
from keras.models import load_model
from keras.utils import pad_sequences


class SplackModel:
    def __init__(self, model_dir, version):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.version = version

    def load(self):
        # Load tokenizer
        with open(f'{self.model_dir}/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        # Load model
        self.model = load_model(f"{self.model_dir}/trained_model")

    def predict(self, query):
        # Now you can use the tokenizer and model for prediction
        sequence = self.tokenizer.texts_to_sequences([query])
        padded_sequence = pad_sequences(sequence, maxlen=150)
        prediction = self.model.predict(padded_sequence)
        return prediction


class S010Prompt(SplackModel):
    def __init__(self):
        super().__init__('../010_prompt', '010_prompt_005_007')


class S011More(SplackModel):
    def __init__(self):
        super().__init__('../011_more', '011_more_005_007')
