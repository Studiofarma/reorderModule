#%%
import numpy as np
import matplotlib.pyplot as plt
from common import data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

tf.debugging.set_log_device_placement(False)
tf.config.set_visible_devices([], 'GPU')

def root_mean_squared_error(y, y_pred):
    return math.sqrt(mean_squared_error(y, y_pred))

def wave(frequency, resolution, amplitude, fn=np.sin):
    length = np.pi * 2 * frequency
    my_wave = fn(np.arange(0, length, length / resolution))
    return my_wave * amplitude

def noise(resolution, amplitude):
    return np.random.normal(0, amplitude, resolution)

def plot_signal(signal, title):
    _, _, psd = data.fft_denoiser_series(signal, 0.5, signal.index)
    data.scatter_lines([signal], title = title)
    data.scatter_lines([abs(psd)])
    
def scale_fn(signal, scaler, tranform_fn):
    scaled = tranform_fn(scaler, signal.values.reshape(-1, 1))
    scaled = pd.Series(scaled.reshape(-1))
    return scaled, scaler

def scale(signal, scaler):
    return scale_fn(signal, scaler, lambda sc, si: sc.fit_transform(si))
    
def unscale(signal, scaler):
    return scale_fn(signal, scaler, lambda sc, si: sc.inverse_transform(si))

def unscale_np(signal, scaler):
    return unscale(pd.Series(signal), scaler)

def generate_scaleds(signal):
    signal_min_max, scaler_min_max = scale(signal, MinMaxScaler())
    signal_standard, scaler_standard = scale(signal, StandardScaler())
    signal_robust, scaler_robust = scale(signal, RobustScaler())

    data.scatter_lines([signal, signal_min_max, signal_standard, signal_robust])
    
    return {
        'signal_min_max': signal_min_max, 
        'scaler_min_max': scaler_min_max,
        'signal_standard': signal_standard, 
        'scaler_standard': scaler_standard,
        'signal_robust': signal_robust, 
        'scaler_robust': scaler_robust
    }
    
def plot_and_scale_signal(signal, title):
    plot_signal(signal, 'seasonal high noisy')
    return generate_scaleds(signal)

resolution = 500
noise_val = noise(resolution, 3)
signal_seasonal = pd.Series(
    wave(5, resolution, 3, np.cos)) + \
    wave(1, resolution, 5, np.cos) + \
    wave(1, resolution, 6, np.sin) + \
    wave(2, resolution, 7, np.sin) + \
    wave(30, resolution, 2, np.cos) + \
    noise_val    
scaled_seasonal = plot_and_scale_signal(signal_seasonal, 'seasonal high noisy')

signal = scaled_seasonal['signal_min_max']
scaler = scaled_seasonal['scaler_min_max']

#%%
def prepare_data_lags(data, lags):
    range_right = lags + 1
    lags_data = {f'y - {range_right - i}':data.shift(range_right - i) for i in range(1, range_right)}
    df = pd.DataFrame(lags_data)
    df['y'] = data
    df = df.dropna()
    
    return df, df.values[:, :-1].astype('float32'), df.values[:, -1].astype('float32')

def split_test(data_df, split_percentage=0.8, smooth=False):
    train_len = int(len(data_df) * split_percentage)
    train_df = data_df.loc[:train_len]
    if smooth:
        train_df = train_df.rolling(window=5).mean().dropna()
    test_df = data_df.loc[train_len:]
    
    return train_df.values[:, :-1], train_df.values[:, -1], test_df.values[:, :-1], test_df.values[:, -1], train_df, test_df

signal_df, _, _ = prepare_data_lags(signal, 10)
X_train, y_train, X_test, y_test, train_df, test_df = split_test(signal_df, 0.6)
X_train_smooth, y_train_smooth, X_test, y_test, train_df_smooth, test_df = split_test(signal_df, 0.6, smooth=True)

data.scatter_lines([train_df['y'], train_df_smooth['y'], test_df['y']])

y_test_unscaled, _ = unscale_np(y_test, scaler)

def evaluate(y_test, forecast):
    error = root_mean_squared_error(y_test, forecast)
    print(f'error 1: {error}')
    test_mean = y_test.mean()
    print(f'test mean: {test_mean}')
    data.scatter_lines_np([y_test_unscaled, forecast])

evaluate(y_test_unscaled, [y_test_unscaled.mean()] * len(y_test_unscaled))
evaluate(y_test_unscaled[1:], y_test_unscaled.shift(1).dropna())

from tensorflow import keras
from keras import layers, activations

def run_model(model_layers, X_train, y_train, X_test, y_test, epochs, batch_size=32, name=''):
    model = keras.models.clone_model(keras.Sequential(model_layers, name=name))
    model.compile(optimizer='Adam', loss='mean_squared_error')

    history = model.fit(
        X_train, 
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=True)

    error = model.evaluate(X_test, y_test, verbose=True)
    forecast = model.predict(X_test)
    forecast, _ = unscale_np(forecast.reshape(-1), scaler)
    y_test_unscaled, _ = unscale_np(y_test, scaler)
    
    return forecast, y_test_unscaled, error, history, model


def evaluate_model(run_times, model_layers, X_train, y_train, X_test, y_test, epochs, batch_size=32, name=''):
    errors = []
    forecasts = []
    models = []
    y_test_unscaled, _ = unscale_np(y_test, scaler)
    
    for i in range(run_times):
        forecast, _, error, history, model = run_model(
            model_layers, 
            X_train, y_train, X_test, y_test,
            epochs=epochs, 
            batch_size=batch_size,
            name=name
        )

        error = root_mean_squared_error(y_test_unscaled, forecast)
        errors.append(error)
        forecasts.append(forecast)
        models.append(model)
        data.scatter_lines([pd.Series(history.history['loss']), pd.Series(history.history['val_loss'])])

    test_mean = y_test_unscaled.mean()
    print(f'errors: {errors}')
    print(f'mean error: {np.array(errors).mean()}')
    print(f'test mean: {test_mean}')
    models[0].summary()
    
    data.scatter_lines([y_test_unscaled] + forecasts)

# %%
epochs=200
batch_size=32
evaluate_model(
    5,
    [
        layers.SimpleRNN(X_train.shape[1], activation=activations.tanh),
        layers.Dense(5, activation=activations.tanh),
        layers.Dense(1, activation=activations.linear)
    ], 
    X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test,
    epochs=epochs,
    batch_size=batch_size,
    name=f'rnn_epochs_{epochs}_batch_size_{batch_size}'
)

epochs=200
batch_size=32
evaluate_model(
    5,
    [
        layers.LSTM(X_train.shape[1], activation=activations.tanh),
        layers.Dense(5, activation=activations.tanh),
        layers.Dense(1, activation=activations.linear)
    ], 
    X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test,
    epochs=epochs,
    batch_size=batch_size,
    name=f'LSTM_epochs_{epochs}_batch_size_{batch_size}'
)

epochs=200
batch_size=32
evaluate_model(
    5,
    [
        layers.GRU(X_train.shape[1], activation=activations.tanh),
        layers.Dense(5, activation=activations.tanh),
        layers.Dense(1, activation=activations.linear)
    ], 
    X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test,
    epochs=epochs,
    batch_size=batch_size,
    name=f'GRU_epochs_{epochs}_batch_size_{batch_size}'
)

epochs=200
batch_size=32
evaluate_model(
    5,
    [
        layers.Dense(X_train.shape[1], activation=activations.relu),
        layers.Dense(1, activation=activations.linear)
    ], 
    X_train, y_train, X_test, y_test,
    epochs=epochs,
    batch_size=batch_size,
    name=f'single_layer_epochs_{epochs}_batch_size_{batch_size}'
)

epochs=300
batch_size=32
evaluate_model(
    5,
    [
        layers.Dense(4, activation=activations.relu),
        layers.Dense(8, activation=activations.relu),
        layers.Dense(3, activation=activations.relu),
        layers.Dense(1, activation=activations.linear)
    ], 
    X_train, y_train, X_test, y_test,
    epochs=epochs,
    batch_size=batch_size,
    name=f'deep_epochs_{epochs}_batch_size_{batch_size}'
)

epochs=300
batch_size=32
evaluate_model(
    5,
    [
        layers.Dense(X_train.shape[1], activation=activations.relu),
        layers.Dense(8, activation=activations.relu),
        layers.Dense(3, activation=activations.relu),
        layers.Dense(1, activation=activations.linear)
    ], 
    X_train, y_train, X_test, y_test,
    epochs=epochs,
    batch_size=batch_size,
    name=f'deep_epochs_{epochs}_batch_size_{batch_size}'
)

#%%
epochs=200
batch_size=32
evaluate_model(
    5,
    [
        layers.Conv1D(64, kernel_size=3, activation=activations.relu),
        layers.Flatten(),
        # mettere un dropout?
        layers.Dense(5, activation=activations.relu),
        layers.Dense(1, activation=activations.linear)
    ], 
    X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test,
    epochs=epochs,
    batch_size=batch_size,
    name=f'conv1D_layer_epochs_{epochs}_batch_size_{batch_size}'
)

epochs=200
batch_size=32
evaluate_model(
    5,
    [
        layers.Conv1D(64, kernel_size=3, activation=activations.relu),
        layers.Flatten(),
        layers.Dropout(0.1),
        layers.Dense(5, activation=activations.relu),
        layers.Dense(1, activation=activations.linear)
    ], 
    X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test,
    epochs=epochs,
    batch_size=batch_size,
    name=f'conv1D_layer_epochs_{epochs}_batch_size_{batch_size}'
)

epochs=200
batch_size=32
evaluate_model(
    5,
    [
        layers.Conv1D(64, kernel_size=3, activation=activations.relu),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(5, activation=activations.relu),
        layers.Dense(1, activation=activations.linear)
    ], 
    X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test,
    epochs=epochs,
    batch_size=batch_size,
    name=f'conv1D_layer_epochs_{epochs}_batch_size_{batch_size}'
)

#%%
epochs=200
batch_size=32
evaluate_model(
    5,
    [
        layers.Conv1D(64, kernel_size=5, activation=activations.relu),
        layers.MaxPooling1D(pool_size=2, strides=2),
        layers.Conv1D(128, kernel_size=3, activation=activations.relu),
        layers.Flatten(),
        layers.Dense(5, activation=activations.relu),
        layers.Dense(1, activation=activations.linear)
    ], 
    X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test,
    epochs=epochs,
    batch_size=batch_size,
    name=f'conv1D_layer_epochs_{epochs}_batch_size_{batch_size}'
)
