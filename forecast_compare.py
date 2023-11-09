#%%
initial_stock=0
minsan = '004763114' # ASPIRINA C*10CPR EFF 400+240MG (TO ma bello perchè stagionale, vedi autocorr)
# minsan = 'A0420030101' # BD Category febbre e raffreddore
# minsan = 'X0144010101' # BD Category farmaco prescritto
# minsan = '024840074' # acquisti molto regolari
# minsan = '036635011' # DIBASE*OS GTT 10ML 10000UI/ML
data_dir='data'

from telnetlib import SE
import tensorflow as tf
tf.debugging.set_log_device_placement(False)
tf.config.set_visible_devices([], 'GPU')
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
from multiprocessing import reduction
import numpy as np
import common.data as data
import pandas as pd
from plotly import express as px
from common import data
from datetime import date, datetime, timedelta
from calendar import monthrange
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

SEED = 0
import os
import random

#Function to initialize seeds for all libraries which might have stochastic behavior
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_global_determinism(seed=SEED)

scaler = MinMaxScaler()

def custom_smoothing(xs: pd.DataFrame):
    kernel = signal.gaussian(9, 1)[:5]
    kernelSum = kernel.sum()
    smoothed = xs.rolling(window = 5).apply(lambda window: (window * kernel).sum() / kernelSum)
    return smoothed

sales_data = data.read_sell(minsan, data_dir, extension='csv')
sales_data = data.leave_work_days(sales_data, work_days=7)

start_date_fill = datetime(2001, 1, 1)
buy_sell_data, weekly_index, monthly_index, quarter_index, quad_index = \
    data.sales_resample(sales_data, start_date_fill)

index = weekly_index
weekly_sales = buy_sell_data.loc[index].avg_weekly_sell_qta
noisy_sales = pd.DataFrame({'y': weekly_sales}, index = index)
avg_sales = pd.DataFrame({'y': buy_sell_data.loc[index].avg_weekly_sell_qta})
# noisy_sales = data.quantile(noisy_sales).y
noisy_sales = noisy_sales.y
# sales = noisy_sales.copy()
sales = custom_smoothing(noisy_sales)
data.scatter_lines([noisy_sales, sales], title=f'sales since {start_date_fill}')

start_train = "2003-01-01"
train_date = '2019-12-31'
test_date = '2020-01-01'
first_month_days = monthrange(start_date_fill.year, start_date_fill.month)[1]
second_month_days = monthrange(start_date_fill.year, start_date_fill.month + 1)[1]
third_month_days = monthrange(start_date_fill.year, start_date_fill.month + 2)[1]
second_month = start_date_fill + timedelta(days=first_month_days)
fourth_month = start_date_fill + timedelta(days=first_month_days + second_month_days + third_month_days)

train_test = pd.DataFrame(sales)
train_test['noisy_y'] = noisy_sales
train_test['lag_1'] = train_test['y'].shift(1)
train_test['lag_2'] = train_test['y'].shift(2)
train_test['lag_52'] = train_test['y'].shift(52)

prev_month_index = train_test.index[train_test.index >= second_month].map(data.prev_period)
train_test.loc[second_month:, 'prev_month'] = buy_sell_data.loc[prev_month_index].avg_monthly_sell_qta.values
train_test.loc[second_month:, 'same_month_lag52'] = train_test['prev_month'].shift(48)

prev_quarter_index = train_test.index[train_test.index >= fourth_month].map(lambda d: data.prev_period(d, 3))
train_test.loc[fourth_month:, 'prev_quarter'] = buy_sell_data.loc[prev_quarter_index].avg_quarter_sell_qta.values
train_test.loc[second_month:, 'same_quarter_lag52'] = train_test['prev_quarter'].shift(39)

train, test = (train_test.loc[start_train:train_date], train_test.loc[test_date:])

def evaluate(data_set, forecast_name):
    label = 'noisy_y'
    test_mean = data_set[label].mean()
    data.scatter_lines([data_set['y'], data_set[label], data_set[forecast_name]], title=f'sales since {test_date}')
    mae = (abs(data_set[label] - data_set[forecast_name]).mean())
    print(f'absolute MAE: {mae}')
    print(f'% MAE: {mae / test_mean}')
    print(f'test mean: {test_mean}')
# %% Wingesfar forecast
def wingesfar_forecast(test, weights):
    forecast = (test[['lag_1', 'lag_2', 'prev_month', 'prev_quarter', 'same_month_lag52', 'same_quarter_lag52']].values * weights).sum(axis=1) / sum(weights)
    return forecast

test['wg_forecast'] = wingesfar_forecast(test, [5, 0.5, 8, 3, -1, 0])
# test['wg_forecast'] = wingesfar_forecast(test, [2, 0, -1, 1, 0, 0])
# test['wg_forecast'] = wingesfar_forecast(test, [1.5, 0, 4, 6, 4, 1.5])
# test['wg_forecast'] = wingesfar_forecast(test, [1, 1, 1, 1, 1, 1])
# test['wg_forecast'] = wingesfar_forecast(test, [1, 0, 0, 0, 0, 0])
# test['wg_forecast'] = wingesfar_forecast(test, [0.51, 0.49, 0.65, 0.48, 0.6, 0.7])
evaluate(test, 'wg_forecast')

# %% Simple NN
from tensorflow import keras
from keras import layers, activations, optimizers

set_global_determinism(seed=SEED)

def simple_nn(train, test):
    model = keras.Sequential([
        layers.Dense(12, input_dim=6, activation=activations.relu),
        # layers.Dropout(rate=0.1),
        # layers.Dense(4, activation=activations.relu),
        # layers.Dropout(rate=0.1),
        layers.Dense(1, activation=activations.linear),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error'
    )

    # features_columns = ['lag_1', 'lag_2', 'prev_month', 'prev_quarter', 'same_month_lag52', 'same_quarter_lag52']
    features_columns = ['lag_1', 'lag_2', 'prev_month', 'prev_quarter', 'same_month_lag52', 'same_quarter_lag52']

    label_columns = ['y']
    train_features = train[features_columns]
    train_labels = train[label_columns]

    test_features = test[features_columns]
    test_labels = test[label_columns]

    history = model.fit(
        train_features.to_numpy().astype('float32'),
        train_labels.to_numpy().astype('float32'),
        # batch_size=5,
        epochs=150,
        validation_data=(test_features, test_labels),
        verbose=True)

    error = model.evaluate(
        test_features.to_numpy().astype('float32'),
        test_labels.to_numpy().astype('float32'),
        verbose=True)

    forecast = model.predict(test_features.to_numpy().astype('float32'))
    print(error)

    model.summary()

    return forecast, history

test['mean'] = test['y'].mean()
evaluate(test, 'mean')

test['nn_forecast'], history = simple_nn(train, test)
evaluate(test, 'nn_forecast')

data.scatter_lines([pd.Series(history.history['loss'])])

# %% Deep NN
import tensorflow as tf
from tensorflow import keras
from keras import layers, activations, optimizers

set_global_determinism(seed=SEED)

def deep_nn(train, test):
    model = keras.Sequential([
        layers.Dense(12, input_dim=6, activation=activations.relu),
        layers.Dense(24, activation=activations.relu),
        layers.Dense(48, activation=activations.relu),
        layers.Dense(24, activation=activations.relu),
        layers.Dropout(rate=0.1),
        layers.Dense(12, activation=activations.relu),
        layers.Dense(1, activation=activations.linear),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error'
    )

    # features_columns = ['lag_1', 'lag_2', 'prev_month', 'prev_quarter', 'same_month_lag52', 'same_quarter_lag52']
    features_columns = ['lag_1', 'lag_2', 'prev_month', 'prev_quarter', 'same_month_lag52', 'same_quarter_lag52']

    label_columns = ['y']
    train_features = train[features_columns]
    train_labels = train[label_columns]

    test_features = test[features_columns]
    test_labels = test[label_columns]

    history = model.fit(
        train_features.to_numpy().astype('float32'),
        train_labels.to_numpy().astype('float32'),
        # batch_size=1,
        epochs=100,
        validation_data=(test_features, test_labels),
        verbose=True)

    error = model.evaluate(
        test_features.to_numpy().astype('float32'),
        test_labels.to_numpy().astype('float32'),
        verbose=True)

    forecast = model.predict(test_features.to_numpy().astype('float32'))
    print(error)

    model.summary()

    return forecast, history

test['mean'] = test['y'].mean()
evaluate(test, 'mean')

test['nn_forecast'], history = deep_nn(train, test)
test['nn_forecast'] = test['nn_forecast']
evaluate(test, 'nn_forecast')

data.scatter_lines([pd.Series(history.history['loss'])])

# %% Recurrent NN
import tensorflow as tf
from tensorflow import keras
from keras import layers, activations, optimizers

set_global_determinism(seed=SEED)

def prep_data(datain, time_step):
    y_indices = np.arange(start=time_step, stop=len(datain), step=time_step)
    y_tmp = datain[y_indices]

    rows_X = len(y_tmp)
    X_tmp = datain[range(time_step*rows_X)]
    X_tmp = np.reshape(X_tmp, (rows_X, time_step, 1))
    return X_tmp, y_tmp

def prep_data2(datain, time_step):
    y_indices = np.arange(start=time_step, stop=len(datain), step=1)
    y_tmp = datain[y_indices]
    X_tmp = np.array([datain[i:time_step + i] for i in range(len(datain) - time_step)])

    return X_tmp, y_tmp

def recurrent_nn(train, test, timestep):
    train_features, train_labels = prep_data2(train, timestep)
    test_data = np.concatenate((train[-timestep:], test))
    test_features, test_labels = prep_data2(test_data, timestep)

    model = keras.Sequential([
        layers.Input(shape = (timestep, 1)),
        # layers.Conv1D(9, 3, strides=1, activation=activations.relu, name="conv2"),
        layers.Dense(units = 4, activation=activations.tanh),
        # layers.GRU(units = 5, recurrent_dropout=0.1, activation=activations.tanh, name="recurrent1", return_sequences=True),
        layers.GRU(units = 5, recurrent_dropout=0.1, activation=activations.tanh, name="recurrent1"),
        # layers.Flatten(),
        layers.BatchNormalization(),
        layers.Dense(units = 4, activation=activations.tanh),
        layers.Dropout(rate=0.1),
        layers.Dense(units = 1, activation=activations.linear),
    ])

    # model = keras.Sequential([
    #     layers.Input(shape = (timestep, 1)),
    #     layers.GRU(units = 3, activation=activations.tanh, name="recurrent1"),
    #     # layers.Dense(units = 2, activation=activations.tanh),
    #     layers.Dense(units = 1, activation=activations.linear),
    # ])

    # model = keras.Sequential([
    #     layers.Input(shape = (timestep, 1)),
    #     layers.SimpleRNN(units = int((timestep / 2)) + 1, activation=activations.tanh, name="recurrent1"),
    #     layers.Dense(units = timestep, activation=activations.tanh),
    #     layers.Dropout(rate=0.1),
    #     layers.Dense(units = int((timestep / 2)) + 1, activation=activations.tanh),
    #     layers.Dropout(rate=0.1),
    #     layers.Dense(units = 2, activation=activations.tanh),
    #     layers.Dense(units = 1, activation=activations.linear),
    # ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error'
    )


    history = model.fit(
        train_features,
        train_labels,
        batch_size=10,
        epochs=150,
        validation_data=(test_features, test_labels),
        verbose=True)

    error = model.evaluate(
        test_features,
        test_labels,
        verbose=True)

    forecast = model.predict(test_features)
    print(error)

    model.summary()

    forecast = forecast.reshape(len(forecast))
    return forecast, history

train_test_rnn = np.concatenate((train['y'].values, test['y'].values))
train_test_rnn = scaler.fit_transform(train_test_rnn.reshape(len(train_test_rnn), 1)).reshape(len(train_test_rnn))
train_test_rnn_mean = train_test_rnn.mean()
train_test_rnn = train_test_rnn - train_test_rnn_mean
train_rnn = train_test_rnn[:len(train)]
test_rnn = train_test_rnn[len(train):]

# nn_forecast_scaled, history = recurrent_nn(train_rnn, test_rnn, 1)
# nn_forecast_scaled = nn_forecast_scaled + train_test_rnn_mean
# test['nn_forecast'] = scaler.inverse_transform(nn_forecast_scaled.reshape(-1, 1)).reshape(-1)
# evaluate(test, 'nn_forecast')

# data.scatter_lines([pd.Series(history.history['loss'])])

# nn_forecast_scaled, history = recurrent_nn(train_rnn, test_rnn, 8)
# nn_forecast_scaled = nn_forecast_scaled + train_test_rnn_mean
# test['nn_forecast'] = scaler.inverse_transform(nn_forecast_scaled.reshape(-1, 1)).reshape(-1)
# evaluate(test, 'nn_forecast')

# data.scatter_lines([pd.Series(history.history['loss'])])

# nn_forecast_scaled, history = recurrent_nn(train_rnn, test_rnn, 5)
# nn_forecast_scaled = nn_forecast_scaled + train_test_rnn_mean
# test['nn_forecast'] = scaler.inverse_transform(nn_forecast_scaled.reshape(-1, 1)).reshape(-1)
# evaluate(test, 'nn_forecast')

# data.scatter_lines([pd.Series(history.history['loss'])])

nn_forecast_scaled, history = recurrent_nn(train_rnn, test_rnn, 3)
nn_forecast_scaled = nn_forecast_scaled + train_test_rnn_mean
test['nn_forecast'] = scaler.inverse_transform(nn_forecast_scaled.reshape(-1, 1)).reshape(-1)
evaluate(test, 'nn_forecast')

data.scatter_lines([pd.Series(history.history['loss'])])

# %% Arima
from statsmodels.tsa.arima.model import ARIMA
def arima(train, test):
    def inner_arima(train, xs):
        step = 10
        arima_model = ARIMA(np.concatenate((train, xs)), order=(1, 1, step))
        model = arima_model.fit()
        return model.forecast(1)[0]

    forecast = [inner_arima(train, test[:i]) for i in range(0, len(test))]

    return forecast

test['nn_forecast'] = arima(train['y']['2019-01-01':].values, test['y'].values)
evaluate(test, 'nn_forecast')

# %%
test['nn_forecast'] = test['y'].shift(1)
evaluate(test, 'nn_forecast')
