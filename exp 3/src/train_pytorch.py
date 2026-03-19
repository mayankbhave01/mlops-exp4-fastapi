#!/usr/bin/env python3
"""PyTorch Model Training Script"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


class NeuralNet(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == "__main__":

    df = pd.read_csv("data/dataset.csv")

    X = df.drop("target", axis=1).values
    y = df["target"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)

    model = NeuralNet(X_train.shape[1], len(np.unique(y)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pytorch_model.pth")

    print("PyTorch model saved!")