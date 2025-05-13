import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense

df = pd.read_csv("datasets/classification_data.txt")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
word_index = tokenizer.word_index

maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=maxlen)

label_index = {label: i for i, label in enumerate(df['label'].unique())}
y = to_categorical([label_index[label] for label in df['label']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Embedding(len(word_index)+1, 16, input_length=maxlen))
model.add(SimpleRNN(32))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
