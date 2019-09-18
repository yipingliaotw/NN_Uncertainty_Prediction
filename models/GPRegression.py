import numpy as np
import matplotlib.pyplot as plt


class GPRegression:
    def __init__(self, rbf_varance, rbf_lengthscale, noise_variance):
        self.rbf_variance = rbf_varance
        self.rbf_lengthscale = rbf_lengthscale
        self.noise_variance = noise_variance

    def rbf_kernel(self, x_p, x_q):
        n_p = x_p.shape[0]
        n_q = x_q.shape[0]
        k = np.zeros((n_p, n_q))
        for i in range(n_p):
            for j in range(n_q):
                k[i, j] = np.sum(self.rbf_variance *
                                 np.exp(-1. / (2 * self.rbf_lengthscale) * (x_p[i] - x_q[j]) ** 2))
        return k

    def draw_sample(self, m, k, number_draws):
        n = k.shape[0]
        L_chol = np.linalg.cholesky(k + 1e-6 * np.eye(n))
        s = np.dot(L_chol, np.random.normal(size=(n, number_draws)))
        s = s.T + m
        return s

    def visualize_prior(self, num_samples=5):
        x = np.linspace(-4, 4, 100)
        cov = self.rbf_kernel(x, x)
        mean = np.zeros(x.shape)
        samples_prior = self.draw_sample(mean, cov, num_samples)

        plt.figure(figsize=(12, 8))
        for i in range(num_samples):
            plt.plot(x, samples_prior[i, :])

        upper = mean + 2 * np.sqrt(np.diag(cov))
        lower = mean - 2 * np.sqrt(np.diag(cov))
        plt.fill_between(x, upper, lower, color='k', alpha=0.2)
        plt.title("prior")
        plt.xlabel("input x")
        plt.ylabel("output f(x)")
        plt.ylim(-20, 20)
        plt.show()

    def basis_function(self, ipt):
        return np.concatenate((np.ones_like(ipt), ipt, ipt**2, ipt**3), axis=1)

    def fit_pred(self, train_input, train_output, test_input):
        kff = self.rbf_kernel(train_input, train_input)
        kfy = self.rbf_kernel(train_input, test_input)
        kyy = self.rbf_kernel(test_input, test_input)
    
        tmp = np.dot(kfy.T, np.linalg.pinv(kff + self.noise_variance*np.eye(kff.shape[0])))
        m = np.dot(tmp, train_output)
        c = kyy - np.dot(tmp, kfy)+ 1e-5*np.eye(kyy.shape[0])
        return m, c
