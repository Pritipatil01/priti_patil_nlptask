from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
import numpy as np

with open("datasets/generation_data.txt") as f:
    text = f.read().lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split('.'):
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        input_sequences.append(tokens[:i+1])

maxlen = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=maxlen, padding='pre')

X = input_sequences[:, :-1]
y = to_categorical(input_sequences[:, -1], num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 16, input_length=maxlen-1))
model.add(SimpleRNN(64))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=50, verbose=1)

# Text generation
seed_text = "science is the study"
next_words = 20

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=maxlen-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    next_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():
        if index == next_index:
            seed_text += " " + word
            break

print(seed_text)
