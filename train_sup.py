import pandas as pd

from dataset import *
from neural_network import *
import torch
from dist import *
import torch.nn as nn
import torch.optim as optim


def train_supervised(sup_model, lr, epochs, X_train, y_train, batch_size):
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
    optimizer = optim.Adam(sup_model.parameters(), lr=lr, weight_decay=0.0001)
    sup_model.train()

    avg_distances = {}
    avg_distances[(0, 0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []
    accuracy_values = []
    correct = 0
    total = 0


    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch_size):
            optimizer.zero_grad()
            inputs = X_train[i:i + batch_size]
            labels = y_train[i:i + batch_size]
            outputs = sup_model(inputs.reshape(inputs.shape[0],1, 18))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            zero, zero_one, one = sampled_all_distance(sup_model.encoded, labels)

            avg_distances[(0, 0)].append(zero)
            avg_distances[(0, 1)].append(zero_one)
            avg_distances[(1, 1)].append(one)
            torch.save(sup_model.state_dict(),
                       "weights/sup/sup_net_weights_" + " lr=" + str(lr) + "_" + str(
                           epoch) + "_" + str(int(i / batch_size)) + ".pth")

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            accuracy_values.append(accuracy)
            print("acc:", accuracy)
            print(zero, zero_one, one)
            print(i, loss.item())

    df = pd.DataFrame(avg_distances)
    df["accuracy"] = accuracy_values
    df.to_csv(f"sup lr={lr}")