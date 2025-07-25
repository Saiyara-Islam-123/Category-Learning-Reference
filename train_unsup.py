from dataset import *
from neural_network import *
import torch
from dist import *
import torch.nn as nn
import torch.optim as optim


def train_unsupervised(unsup_model, lr, epochs, train_set, test_set, batch_size):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(unsup_model.parameters(), lr=lr)
    unsup_model.train()

    avg_distances = {}
    avg_distances[(0, 0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []
    accuracies = []

    for epoch in range(epochs):
        for i in range(0, len(train_set), batch_size):
            optimizer.zero_grad()
            inputs = train_set[i:i + batch_size]
            labels = inputs[:, -1]
            outputs = unsup_model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            zero, zero_one, one = sampled_all_distance(unsup_model.encoded, labels)

            avg_distances[(0, 0)].append(zero)
            avg_distances[(0, 1)].append(zero_one)
            avg_distances[(1, 1)].append(one)
            torch.save(unsup_model.state_dict(),
                       "../net_weights/unsup/unsup_net_weights_" + " lr=" + str(lr) + "_" + str(
                           epoch) + "_" + str(i) + ".pth")

            print(i, loss.item())

            