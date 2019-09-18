import keras
import keras.backend as K
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras.optimizers import Adam


class BayesianNN:
    def __init__(self, num_models, num_epochs, batch_size, task=None):
        self.num_models = num_models
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.task = task

    @staticmethod
    def aleatoric_loss(var):
        def loss(y_true, y_pred):
            diff = (y_true - y_pred) ** 2
            loss_value = 0.5 * K.exp(-var) * diff + 0.5 * var
            loss_value = K.mean(loss_value)
            return loss_value
        return loss

    @staticmethod
    def mse():
        def loss(y_true, y_pred):
            diff = (y_true - y_pred) ** 2
            loss_value = K.mean(diff)
            return loss_value
        return loss

    def model_dropout_var(self):
        ipt = Input((1,))
        layer = Dense(512, activation='elu')(ipt)
        layer = Dropout(0.2)(layer)
        layer = Dense(512, activation='elu')(layer)
        layer = Dropout(0.2)(layer)
        layer = Dense(512, activation='elu')(layer)
        layer = Dropout(0.2)(layer)
        var = Dense(1, activation=None, name='var')(layer)
        pred = Dense(1, activation=None, name='pred')(layer)
        model = Model(ipt, pred)
        model.summary()
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        if self.task is "model_data_uncertainty":
            model.compile(optimizer=adam, loss=self.aleatoric_loss(var))
        elif self.task is "model_uncertainty":
            model.compile(optimizer=adam, loss="mse")

        get_variance = K.function([ipt, K.learning_phase()], [var])
        get_pred = K.function([ipt, K.learning_phase()], [pred])
        return model, get_pred, get_variance

    def model_var(self):
        ipt = Input((1,))
        layer = Dense(512, activation='elu')(ipt)
        layer = Dense(512, activation='elu')(layer)
        layer = Dense(512, activation='elu')(layer)
        var = Dense(1, activation=None, name='var')(layer)
        pred = Dense(1, activation=None, name='pred')(layer)
        model = Model(ipt, pred)
        model.summary()

        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(optimizer=adam, loss=self.aleatoric_loss(var))
        get_variance = K.function([ipt, K.learning_phase()], [var])
        get_pred = K.function([ipt, K.learning_phase()], [pred])
        return model, get_pred, get_variance

    def get_model(self):
        if self.task is "model_data_uncertainty":
            return self.model_dropout_var()
        elif self.task is "model_uncertainty":
            return self.model_dropout_var()
        elif self.task is "data_uncertainty":
            return self.model_var()

    def fit_pred(self, train_input, train_output, test_input):
        model, get_pred, get_variance = self.get_model()
        model.fit(train_input, train_output, batch_size=self.batch_size,
                  epochs=self.num_epochs, validation_split=0.2)

        for i in range(self.num_models):
            pred = get_pred([test_input, 1])[0]
            log_var = get_variance([test_input, 1])[0]
            if i == 0:
                pred_array = np.copy(pred)
                log_var_array = np.copy(log_var)
            else:
                pred_array = np.concatenate((pred_array, pred), axis=1)
                log_var_array = np.concatenate((log_var_array, log_var), axis=1)

        var = np.exp(log_var_array)

        pred_mean = np.mean(pred_array, axis=1)
        model_var = np.var(pred_array, axis=1)
        data_var = np.mean(var, axis=1)

        if self.task is "model_uncertainty":
            data_var = 0
        elif self.task is "data_uncertainty":
            model_var = 0
        pred_var = model_var + data_var
        return pred_mean, pred_var
