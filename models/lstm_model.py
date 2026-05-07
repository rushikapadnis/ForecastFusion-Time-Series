import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input

def train_lstm(X_train, y_train):

    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)

    X_train = X_train.reshape(
        (X_train.shape[0], X_train.shape[1], 1)
    )

    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=16,
        verbose=0
    )

    return model
