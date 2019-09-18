import matplotlib.pyplot as plt
import numpy as np


def plot(mean, var, train_x, train_y, test_x, title=None):
    """plot 95% of confidence level"""
    plt.figure(figsize=(12, 8))
    plt.plot(train_x, train_y, 'bx', color="#FF8247", alpha=0.3, label="train_data")
    if len(mean.shape) >= 2:
        mean = mean[:, 0]
    if len(var.shape) >= 2:
        var = var[:, 0]

    plt.plot(test_x, mean, color='#104E8B', label="prediction")
    plt.fill_between(test_x, mean - 2 * np.sqrt(var),
                     mean + 2 * np.sqrt(var), color='#4876FF', alpha=0.9, label="95% confidence")

    true = 10 * np.sin(test_x)
    plt.plot(test_x, true, color='#FF8247', label='true')

    if title:
        plt.title(title)
    plt.legend()
