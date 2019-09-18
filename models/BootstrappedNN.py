import keras
import keras.backend as K
import numpy as np

from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras.optimizers import Adam


class BootstrappedNN:
    def __init__(self, num_models, num_epochs, batch_size, bootstrapped_size):
        self.num_models = num_models
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.bootstrapped_size = bootstrapped_size

    @staticmethod
    def get_model():
        ipt = Input((1,))
        layer = Dense(64, activation='elu', name="shared")(ipt)
        layer = Dense(32, activation='elu')(layer)
        pred = Dense(1, activation=None, name='pred')(layer)
        model = Model(ipt, pred)
        model.summary()
        return model

    @staticmethod
    def reset_weights(m):
        session = K.get_session()
        m.get_layer("shared").kernel.initializer.run(session=session)
        return m

    def bootstrap(self, train_x, train_y):
        model_list = []
        for i in range(self.num_models):
            if i == 0:
                model_shared = self.get_model()
                model = keras.models.clone_model(model_shared)
                model.set_weights(model_shared.get_weights())
            else:
                model = self.reset_weights(model_shared)
            model.compile(optimizer='adam', loss="mse")
            idx = np.random.randint(
                train_x.shape[0], size=self.bootstrapped_size)
            train_x_sub = train_x[idx]
            train_y_sub = train_y[idx]
            model.fit(train_x_sub, train_y_sub, batch_size=self.batch_size,
                      epochs=self.num_epochs, validation_split=0.2)

            model_shared = keras.models.clone_model(model)
            model_shared.set_weights(model.get_weights())
            model_list.append(model)
        return model_list

    @staticmethod
    def pred_bootstrap(m_list, test_x):
        pred_list = []
        for m in m_list:
            pred = m.predict(test_x)
            pred_list.append(pred)
            pred_array = np.array(pred_list)
        pred_mean = np.mean(pred_array, axis=0)
        pred_var = np.var(pred_array, axis=0)
        return pred_mean, pred_var

    def fit_pred(self, train_input, train_output, test_input):
        model_list = self.bootstrap(train_input, train_output)
        pred_mean, pred_var = self.pred_bootstrap(model_list, test_input)
        return pred_mean, pred_var
