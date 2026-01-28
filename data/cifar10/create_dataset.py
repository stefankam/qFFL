import argparse
import json
import os

import numpy as np
from tensorflow.keras.datasets import cifar10


def normalize_data(x_train, x_test):
    mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    return x_train, x_test


def dirichlet_split(labels, num_clients, alpha, rng):
    num_classes = int(np.max(labels)) + 1
    client_indices = [[] for _ in range(num_clients)]
    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(cls_indices)).astype(int)
        cls_split = np.split(cls_indices, splits[:-1])
        for client_id, split in enumerate(cls_split):
            client_indices[client_id].extend(split.tolist())
    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])
    return client_indices


def build_user_data(x, y, indices):
    user_x = x[indices]
    user_y = y[indices]
    return user_x.tolist(), user_y.tolist()


def write_data_json(output_path, users, user_data):
    data = {
        'users': users,
        'user_data': user_data,
        'num_samples': [len(user_data[user]['y']) for user in users],
    }
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./data')
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train, x_test = normalize_data(x_train, x_test)

    train_client_indices = dirichlet_split(y_train, args.num_clients, args.alpha, rng)
    test_client_indices = dirichlet_split(y_test, args.num_clients, args.alpha, rng)

    train_users = []
    test_users = []
    train_user_data = {}
    test_user_data = {}

    for client_id in range(args.num_clients):
        user = 'client_{:05d}'.format(client_id)
        train_users.append(user)
        test_users.append(user)
        train_x, train_y = build_user_data(x_train, y_train, train_client_indices[client_id])
        test_x, test_y = build_user_data(x_test, y_test, test_client_indices[client_id])
        train_user_data[user] = {'x': train_x, 'y': train_y}
        test_user_data[user] = {'x': test_x, 'y': test_y}

    train_output_dir = os.path.join(args.output_dir, 'train')
    test_output_dir = os.path.join(args.output_dir, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    write_data_json(os.path.join(train_output_dir, 'cifar10_train.json'), train_users, train_user_data)
    write_data_json(os.path.join(test_output_dir, 'cifar10_test.json'), test_users, test_user_data)


if __name__ == '__main__':
    main()
