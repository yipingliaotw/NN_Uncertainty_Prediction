import keras
import keras.backend as K
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam


class EnsembledNN:
    def __init__(self, num_models, num_epochs, batch_size):
        self.num_models = num_models
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    @staticmethod
    def get_model():
        ipt = Input((1,))
        layer = Dense(64, activation='relu')(ipt)
        layer = Dense(32, activation='relu')(layer)
        pred = Dense(1, activation=None, name='pred')(layer)
        model = Model(ipt, pred)
        adam = keras.optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(optimizer=adam, loss="mse")
        model.summary()
        return model

    def ensemble(self, train_x, train_y):
        model_list = []
        for i in range(self.num_models):
            model = self.get_model()
            idx = np.random.permutation(train_x.shape[0])
            train_x_sub = train_x[idx]
            train_y_sub = train_y[idx]
            model.fit(train_x_sub, train_y_sub, batch_size=self.batch_size,
                      epochs=self.num_epochs, validation_split=0.2)
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
