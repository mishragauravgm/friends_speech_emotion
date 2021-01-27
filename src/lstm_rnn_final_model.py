from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras.optimizers import RMSprop, Adam, SGD

PATH = '/content/drive/My Drive/Midas_task'

import warnings
warnings.filterwarnings("ignore")

UNITS = 500 #v1 (500, too slow), v2(100,too slow)
BS = 16
EPOCHS = 20
OPT = Adam(lr=0.001)

model=Sequential();
#model.add(LSTM(UNITS, activation='tanh', recurrent_activation='sigmoid',input_shape=(1,4999)))
model.add(SimpleRNN(UNITS, activation='tanh',input_shape=(1,4999)))
model.add(Dense(5,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer = OPT, metrics=['categorical_accuracy'])

history = model.fit(train_data.values.reshape((6367,1,4999)),train_labels,batch_size=BS, epochs = EPOCHS, validation_data=(val_data.values.reshape((830,1,4999)),val_labels))

model.save('/content/drive/My Drive/Midas_task/mfcc_nceps13/modelRNN.h5')
model.save_weights('/content/drive/My Drive/Midas_task/mfcc_nceps13/modelRNNWeights.h5')