# %%
import numpy as np
from common import data
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import Scatter
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def plot_px(series: list, labels: list, title = ''):
    fig = go.Figure(layout_title_text=title)
    for i, data in enumerate(series):
        data = pd.Series(data)
        fig = fig.add_trace(Scatter(x = data.index, y = data, mode='lines', name=f'{labels[i]}'))
    fig.show()

def root_mean_squared_error(y, y_pred):
    return math.sqrt(mean_squared_error(y, y_pred))

def evaluate_px(y_test, forecast):
    error = root_mean_squared_error(y_test, forecast)
    plot_px([y_test, forecast], ['actual', 'predicted'], title=f'error: {error}')


def baseline(original_series: pd.Series, train_len: int):
    original_series = original_series[train_len:]
    series = pd.Series(original_series).shift(1)
    month = series.rolling(window=4).mean()
    quarter = series.rolling(window=12).mean()

    weights = [1, 1, 1]
    wg_features = pd.DataFrame({'week':series, 'month': month, 'quarter': quarter}).dropna()
    wg_forecast =(wg_features.values * weights).sum(axis=1) / sum(weights)
    evaluate_px(original_series[12:], pd.Series(wg_forecast, index = original_series[12:].index))


initial_stock=0
minsan = '004763114' # ASPIRINA C*10CPR EFF 400+240MG (TO ma bello perchè stagionale, vedi autocorr)
# minsan = '011782012'
data_dir='data'

import datetime
start_date_fill = datetime.date(2003, 1, 1)
work_days=7

sales_data = data.read_sell(minsan, data_dir, extension='csv')
sales_data_df, weekly_index, monthly_index, quarter_index, quad_index = \
    data.sales_resample(sales_data, start_date_fill, work_days = work_days)
weekly_sales_data = sales_data_df.loc[weekly_index].avg_weekly_sell_qta

lags=12
series = weekly_sales_data
scaler = MinMaxScaler(feature_range=(0, 1))
test_perc = 0.7
len_series = len(series)


baseline(series,  len_series - int(len_series * (1 - test_perc)))
