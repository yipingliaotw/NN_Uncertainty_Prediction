import numpy as np


def gen_data():
    data_sigma1 = 3
    data_sigma2 = 1

    data_1 = np.arange(-3, 0, 0.01)[:, np.newaxis]
    y_1 = np.zeros([data_1.shape[0], 1])
    for i in range(data_1.shape[0]):
        y_1[i, 0] = 10 * np.sin(data_1[i, 0]) + np.random.normal(0, data_sigma1)

    data_2 = np.arange(0, 2, 0.001)[:, np.newaxis]
    y_2 = np.zeros([data_2.shape[0], 1])
    for i in range(data_2.shape[0]):
        y_2[i, 0] = 10 * np.sin(data_2[i, 0]) + np.random.normal(0, data_sigma2)

    data_3 = np.arange(1.5, 3, 0.01)[:, np.newaxis]
    y_3 = np.zeros([data_3.shape[0], 1])
    for i in range(data_3.shape[0]):
        y_3[i, 0] = 10 * np.sin(data_3[i, 0]) + np.random.normal(0, data_sigma2)

    data_x = np.concatenate([data_1, data_2, data_3], axis=0)
    data_y = np.concatenate([y_1, y_2, y_3], axis=0)

    data_y_true = np.zeros([data_x.shape[0], 1])
    for i in range(data_x.shape[0]):
        data_y_true[i, 0] = 10 * np.sin(data_x[i, 0])

    idx = np.random.permutation(data_x.shape[0])
    data_x = data_x[idx]
    data_y = data_y[idx]

    return data_x, data_y