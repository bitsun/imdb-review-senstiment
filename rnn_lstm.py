from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.layers import SimpleRNN,LSTM, Convolution1D, Flatten, Dropout
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import pandas as pd
import sys
from keras.utils.vis_utils import plot_model
import numpy as np

embeddings_index = {}
with open("F:\\Models\\glove.6B.200d.txt",encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))


data = pd.read_csv('F:\\Dataset\\aclImdb\\data.cvs',sep='\t',encoding = "ISO-8859-1",on_bad_lines='skip') # tsv file
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(data['text'].values)

num_tokens = tokenizer.num_words+2
embedding_dim = 200
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
    if i>5000:
        break
print("Converted %d words (%d misses)" % (hits, misses))

X = tokenizer.texts_to_sequences(data['text'].values)
X = tf.keras.preprocessing.sequence.pad_sequences(X,maxlen=500)
Y = pd.get_dummies(data['label']).values

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 36) 

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

model = Sequential()
#model.add(Embedding(max_words, embedding_dim, input_length=X.shape[1]))
model.add(embedding_layer)
model.add(LSTM(32,return_sequences=False))
model.add(Dense(2,activation='sigmoid'))

# Model inspection
model.summary()
#plot_model(model, to_file='model.png')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=64,shuffle=True,validation_data=[X_test,y_test])

# Evaluation on the test set
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
