import pandas as pd
import pickle
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM




others_list=pd.read_pickle('others.pickle')
traffic_data_list=pd.read_pickle('traffic_data_list.pickle')
data = np.array(others_list+traffic_data_list)
labelslist = []
for i in range(0,2883):
    labelslist.append(0)
for i in range(2883,5766):
    labelslist.append(1)

labels = np.array(labelslist)

maxlist = []
for list in others_list+traffic_data_list:
    maxlist.append(max(list))
max = max(maxlist)

np.random.seed(12)
print(data.shape[0])

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

training_samples = int(len(indices) * .8)
validation_samples = len(indices) - training_samples

X_train = data[:training_samples]
y_train = labels[:training_samples]
X_valid = data[training_samples: training_samples + validation_samples]
y_valid = labels[training_samples: training_samples + validation_samples]

print(X_valid)

num_events = 63

embedding_dim = 20

embedding_matrix = np.random.rand(num_events, embedding_dim)

units = 32

model = Sequential()
model.add(Embedding(num_events, embedding_dim))
model.add(LSTM(units))
model.add(Dense(1, activation='sigmoid'))

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_valid, y_valid))
model.save("untrainablemodel.h5")


