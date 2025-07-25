import numpy as np

import random
np.random.seed(0)
random.seed(0)
import torch

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
    return torch.tensor(np.array(l))

def get_dataset(vec_size, dataset_size):
    ones = generate_data(cat=1, vec_size=vec_size, dataset_size=dataset_size)
    zeros = generate_data(cat=0, vec_size=vec_size, dataset_size=dataset_size)
    dataset = torch.cat((ones, zeros))
    train_set, test_set = torch.utils.data.random_split(dataset, [0.9, 0.1])

    return train_set, test_set
    #last index for each vector is also the label

if __name__ == '__main__':
    print(create_prototype(num_charac_features=2, vec_size=5))