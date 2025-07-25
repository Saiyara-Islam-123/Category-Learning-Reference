import numpy as np
from sklearn.model_selection import train_test_split
import random
np.random.seed(0)
random.seed(0)
import torch
torch.manual_seed(0)

def create_prototype(num_charac_features, vec_size):
    cat_a_proto = np.zeros(vec_size)
    cat_b_proto = np.zeros(vec_size)

    feature_indices = np.random.choice(vec_size, num_charac_features)

    for i in range(len(feature_indices)):
        prob_cat_a_is_has_one = random.random()
        if prob_cat_a_is_has_one > 0.5:
            cat_a_proto[i] = 1
            cat_b_proto[i] = 0
        else:
            cat_b_proto[i] = 0
            cat_a_proto[i] = 1

    return cat_a_proto, cat_b_proto, feature_indices

def generate_data(cat, vec_size, num_charac_features, dataset_size):
    l = []

    cat_1_proto, cat_2_proto, feature_indices = create_prototype(num_charac_features, vec_size)


    for i in range(dataset_size):
        num_indices = random.randint(0, vec_size-num_charac_features) #num indices to flip
        indices = np.random.choice(vec_size-num_charac_features, num_indices, replace=False)

        cat_1 = cat_1_proto.copy()
        cat_2 = cat_2_proto.copy()

        for index in indices:

            if cat == 1:
                cat_1[index] = 1

            else:
                cat_2[index] = 1

        if cat == 1:
            l.append(cat_1)

        else:
            l.append(cat_2)
    return torch.tensor(np.array(l), dtype=torch.float32)

def get_dataset(vec_size, dataset_size, num_charac_features):
    ones = generate_data(cat=1, vec_size=vec_size, dataset_size=dataset_size, num_charac_features=num_charac_features)
    zeros = generate_data(cat=0, vec_size=vec_size, dataset_size=dataset_size, num_charac_features=num_charac_features)
    labels = []
    for i in range(ones.shape[0]):
        labels.append(1)
    for i in range(zeros.shape[0]):
        labels.append(0)

    labels = torch.tensor(labels, dtype=torch.int64)
    dataset = torch.cat((ones, zeros))

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, labels, test_size=0.2)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    print(get_dataset(vec_size=5, dataset_size=50, num_charac_features=2))
