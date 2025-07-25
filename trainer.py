from train_unsup import *
from train_sup import *

def train():
    X_train, X_test, y_train, y_test = get_dataset(vec_size=18, dataset_size=1200, num_charac_features=3)
    print("Created dataset")
    unsup_model = AutoEncoder()
    train_unsupervised(unsup_model, lr=0.0001, epochs=1, X_train=X_train, y_train= y_train, batch_size=100)
    sup_model = LastLayer(unsup_model)
    train_supervised(sup_model, lr=0.0001, epochs=1, X_train=X_train, y_train= y_train, batch_size=100)

if __name__ == '__main__':
    train()