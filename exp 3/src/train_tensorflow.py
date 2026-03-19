#!/usr/bin/env python3
"""TensorFlow/Keras Model Training Script"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    df = pd.read_csv('data/dataset.csv')

    X = df.drop('target', axis=1).values
    y = df['target'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model(X_train.shape[1], len(np.unique(y)))

    model.fit(X_train, y_train, epochs=20, batch_size=32)

    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", accuracy)

    os.makedirs('models', exist_ok=True)
    model.save('models/tensorflow_model.keras')