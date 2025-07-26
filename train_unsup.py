from dataset import *
from neural_network import *
import torch
from dist import *
import torch.nn as nn
import torch.optim as optim
import pandas as pd



def train_unsupervised(unsup_model, lr, epochs, X_train, y_train, batch_size):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(unsup_model.parameters(), lr=lr)
    unsup_model.train()

    avg_distances = {}
    avg_distances[(0, 0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []


    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch_size):
            optimizer.zero_grad()
            inputs = X_train[i:i + batch_size]
            labels = y_train[i:i + batch_size]
            outputs = unsup_model(inputs.reshape(inputs.shape[0],1, 50))
            loss = criterion(outputs, inputs.reshape(inputs.shape[0],1, 50))
            loss.backward()
            optimizer.step()


            zero, zero_one, one = sampled_all_distance(unsup_model.encoded, labels)
            print(unsup_model.encoded)

            avg_distances[(0, 0)].append(zero)
            avg_distances[(0, 1)].append(zero_one)
            avg_distances[(1, 1)].append(one)
            '''
            torch.save(unsup_model.state_dict(),
                       "weights/unsup/unsup_net_weights_" + " lr=" + str(lr) + "_" + str(
                           epoch) + "_" + str(int(i / batch_size)) + ".pth")
            '''
            print(i/batch_size, loss.item())


            print(zero, zero_one, one)

    df = pd.DataFrame(avg_distances)
    df.to_csv(f"unsup lr={lr}")
