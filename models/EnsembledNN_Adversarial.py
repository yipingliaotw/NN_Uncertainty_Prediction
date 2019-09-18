import keras
import keras.backend as K
import numpy as np

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam


class EnsembledNN_Adversarial:
    def __init__(self, num_models, num_epochs, batch_size, adversarial_eps):
        self.num_models = num_models
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.adversarial_eps = adversarial_eps

    @staticmethod
    def get_model():
        ipt = Input((1,))
        layer = Dense(64, activation='relu')(ipt)
        layer = Dense(32, activation='relu')(layer)
        pred = Dense(1, activation=None, name='pred')(layer)
        y_true = K.placeholder(shape=(None, 1))
        loss = keras.losses.mean_squared_error(y_true, pred)
        get_grad_fun = K.function([ipt, y_true, K.learning_phase()], K.gradients(loss, ipt))
        model = Model(ipt, pred)
        adam = keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(optimizer=adam, loss="mse")
        model.summary()
        return model, get_grad_fun

    def fit_on_batch(self, train_x, train_y):
        model, get_grad = self.get_model()
        print(model)
        idx = np.random.permutation(train_x.shape[0])
        train_size = int(idx.shape[0] * 0.8)
        idx_train = idx[:train_size]
        idx_test = idx[train_size:]
        train_x_sub = train_x[idx_train]
        train_y_sub = train_y[idx_train]
        test_x_sub = train_x[idx_test]
        test_y_sub = train_y[idx_test]
        num_batches = int(np.ceil(train_x_sub.shape[0] / float(self.batch_size)))

        for e in range(self.num_epochs):
            for b in range(num_batches):
                start_idx = b * self.batch_size
                end_idx = min((b + 1) * self.batch_size, train_x_sub.shape[0])
                grad = get_grad([train_x_sub[start_idx:end_idx], train_y_sub[start_idx:end_idx], 1])[0]
                train_adv = train_x_sub[start_idx:end_idx] + self.adversarial_eps * np.sign(grad)
                train_ipt = np.concatenate([train_x_sub[start_idx:end_idx], train_adv], axis=0)
                train_opt = np.concatenate([train_y_sub[start_idx:end_idx], train_y_sub[start_idx:end_idx]], axis=0)
                train_loss = model.train_on_batch(train_ipt, train_opt)
                test_loss = model.test_on_batch(test_x_sub, test_y_sub)
            print("***epoch:{:3d} ** train loss:{:.3f} ** val loss:{:.3f}***".format(e, train_loss, test_loss))
        return model

    def ensemble(self, train_x, train_y):
        model_list = []
        for i in range(self.num_models):
            model = self.fit_on_batch(train_x, train_y)
            model_list.append(model)
        return model_list

    @staticmethod
    def pred_ensemble(m_list, test_x):
        pred_list = []
        for m in m_list:
            pred = m.predict(test_x)
            pred_list.append(pred)
            pred_array = np.array(pred_list)
        pred_mean = np.mean(pred_array, axis=0)
        pred_var = np.var(pred_array, axis=0)
        return pred_mean, pred_var

    def fit_pred(self, train_input, train_output, test_input):

        model_list = self.ensemble(train_input, train_output)
        pred_mean, pred_var = self.pred_ensemble(model_list, test_input)

        return pred_mean, pred_var
