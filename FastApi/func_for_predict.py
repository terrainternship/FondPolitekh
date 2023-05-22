from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError, Huber
from keras.metrics import MeanAbsoluteError
import numpy as np


def dataset_to_windows(window_size=6, dataset=None):
    # в качестве параметров передается размер окна
    # и имя датасета

    x_train = []
    y_train = []

    for i in range(0, len(dataset) - window_size + 1):

        add_x_train = np.array(dataset[i:185 + window_size + i])
        x_train.append(add_x_train)
        add_y_train = np.array(dataset.iloc[:, 0][
                               i + 185 + window_size: 2 * window_size + i + 185])
        # add_y_train = np.array(dataset.iloc[:,0][i + window_size: window_size + i])

        if len(add_y_train) == window_size:
            y_train.append(add_y_train)

    x_train = x_train[:len(y_train)]
    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train


def compile_and_fit(model, features=None, labels=None,
                    patience=5, max_epochs=15, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')

    model.compile(loss=Huber(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(features,
                        labels,
                        epochs=max_epochs,
                        validation_split=0.1,
                        shuffle=False,
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=1)

    return history
