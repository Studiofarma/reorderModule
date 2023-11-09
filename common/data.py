import string
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs import Scatter
from datetime import datetime, timedelta

def prev_period(date: datetime, months_period: int = 1):
    all_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    months = all_months[::months_period]
    period_idx = int((date.month - 1) / months_period)
    starting_month = months[period_idx]
    return datetime(date.year, starting_month, 1) - timedelta(days = 1)

def scatter_lines(datas, mode = 'lines', title = ''):
    fig = go.Figure(layout_title_text=title)
    for i, data in enumerate(datas):
        fig = fig.add_trace(Scatter(x = data.index, y = data, mode=mode, name=f'data_{i}'))
    fig.show()

def scatter_lines_np(data_np_arrays, mode = 'lines', title = ''):
    data_series = [pd.Series(a) for a in data_np_arrays]
    scatter_lines(data_series, mode, title)

def read_buy(minsan, data_dir, initial_stock = 0, extension = 'csv'):
    buy_data = pd.read_csv(f'{data_dir}/orstd-{minsan}.{extension}', sep=';')
    return process_buy(initial_stock, buy_data)

def process_buy(initial_stock, buy_data):
    buy_data = buy_data[[
        'rec-orstd.orstd-key.orstd-tempo.orstd-data',
        'rec-orstd.orstd-key.orstd-fornitore',
        'rec-orstd.orstd-dati.orstd-qta',
        'rec-orstd.orstd-dati.orstd-qta-cons',
        'rec-orstd.orstd-dati.orstd-costo',
    ]]
    buy_data.rename(columns={
        'rec-orstd.orstd-key.orstd-tempo.orstd-data' : 'date',
        'rec-orstd.orstd-key.orstd-fornitore': 'ws',
        'rec-orstd.orstd-dati.orstd-qta': 'buy_qta',
        'rec-orstd.orstd-dati.orstd-qta-cons': 'buy_delivered',
        'rec-orstd.orstd-dati.orstd-costo': 'buy_cost'
    }, inplace=True, errors='raise')
    buy_data['ws'] = buy_data.ws.astype(str)
    buy_data['ws'] = buy_data.ws.str.rstrip()
    buy_data['ws'] = buy_data.ws.str.lstrip('0')
    # buy_data = buy_data[buy_data['buy_qta'] > 0]
    # buy_data = buy_data[buy_data['buy_qta'] > buy_data['buy_qta'].mean()]
    buy_data.loc[0, 'buy_delivered'] = buy_data.loc[0, 'buy_delivered'] + initial_stock
    buy_data['date'] = pd.to_datetime(buy_data.date, format='%Y%m%d')
    buy_data.sort_values(by='date', inplace=True)
    buy_data.set_index(pd.DatetimeIndex(buy_data.date), inplace=True)

    return buy_data

def read_sell(minsan: string, data_dir: string, extension = 'csv'):
    sales_data = pd.read_csv(f'{data_dir}/statava-{minsan}.{extension}', sep=';')
    return process_sales(sales_data)

def process_sales(sells_data):
    sells_data['vb.vb-dati.vb-qta'] = \
        sells_data['vb.vb-dati.vb-qta'] * \
        sells_data['vb.vb-dati.vb-tipo-v'].apply(lambda x: -1 if '-' in x else 1)

    sells_data = sells_data[~sells_data['vb.vb-dati.vb-tipo-v'].str.contains('r')]

    sells_data['vb.vb-dati.vb-qta'] = \
        sells_data['vb.vb-dati.vb-qta'] + \
        sells_data['vb.vb-dati.vb-dati-sospeso.vb-qta-sosp']

    sells_data = sells_data[['vb.vb-key.vb-data', 'vb.vb-dati.vb-qta']]
    sells_data.rename(columns={
        'vb.vb-key.vb-data' : 'date',
        'vb.vb-dati.vb-qta': 'sell_qta',
    }, inplace=True, errors='raise')
    sells_data['date'] = pd.to_datetime(sells_data.date, format='%Y%m%d')

    return sells_data


def resample(start_date_fill, data, column_name, time_frame = '1D', aggregator = lambda d: d.sum(), reindex_by_date = True):
    if reindex_by_date:
        data.set_index(pd.DatetimeIndex(data.date), inplace=True)
    data = data[data.index > str(start_date_fill)]
    column_names = column_name if type(column_name) is list else [column_name]
    zeros = np.zeros(len(column_names)).reshape(1, len(column_names))
    start = pd.DataFrame(zeros, columns=column_names, index=pd.DatetimeIndex([start_date_fill]))
    data = pd.concat([start, data])
    if time_frame == '3M':
        data = aggregator(data.resample(time_frame, closed='left'))
    else:
        data = aggregator(data.resample(time_frame))
    return data

def days(timedelta):
    try:
        return timedelta.days
    except AttributeError:
        return timedelta

def orders_days_of_lag(orders_dates):
    coupled_dates = map(lambda p: days(p[1] - p[0]), zip(orders_dates, orders_dates[1:]))
    return pd.DataFrame({
        'date' : orders_dates[:-1],
        'other date' : orders_dates[1:],
        'days_lag' : coupled_dates
    }, index=orders_dates[:-1])

def merge_by_day(to_merge, to_be_merged, key, read_key='sell_qta'):
    to_merge_resampled = to_merge[read_key].resample('1D').first().fillna(0)
    resampled_df = pd.DataFrame(to_merge_resampled).rename(columns={read_key: key})
    return pd.merge(to_be_merged, resampled_df, how='outer', left_index=True, right_index=True).fillna(0)

def leave_work_days(df, work_days=6):
    return df.rename_axis("date").query(f"date.dt.dayofweek < {work_days}")

def sales_resample(sells_data, start_date_fill, reindex_by_date = True, work_days=6):
    if leave_work_days:
        sells_data = leave_work_days(sells_data, work_days)
    dayly_sells_data = resample(start_date_fill, sells_data, 'sell_qta', reindex_by_date = reindex_by_date)
    weekly_sells_data = resample(start_date_fill, sells_data, 'sell_qta', time_frame='1W', reindex_by_date = reindex_by_date)
    monthly_sells_data = resample(start_date_fill, sells_data, 'sell_qta', time_frame='1M', reindex_by_date = reindex_by_date)
    quarter_sells_data = resample(start_date_fill, sells_data, 'sell_qta', time_frame='3M', reindex_by_date = reindex_by_date)
    quad_sells_data = resample(start_date_fill, sells_data, 'sell_qta', time_frame='4M', reindex_by_date = reindex_by_date)

    avg_weekly_sells_data = dayly_sells_data.resample('1W').mean()
    avg_monthly_sells_data = dayly_sells_data.resample('1M').mean()
    avg_quarter_sells_data = dayly_sells_data.resample('3M', closed='left').mean()
    avg_quad_sells_data = dayly_sells_data.resample('4M', closed='right').mean()

    sells_dataframe = merge_by_day(weekly_sells_data, dayly_sells_data, 'weekly_sell_qta')
    sells_dataframe = merge_by_day(monthly_sells_data, sells_dataframe, 'monthly_sell_qta')
    sells_dataframe = merge_by_day(quarter_sells_data, sells_dataframe, 'quarter_sell_qta')
    sells_dataframe = merge_by_day(quad_sells_data, sells_dataframe, 'quad_sell_qta')
    sells_dataframe = merge_by_day(avg_weekly_sells_data, sells_dataframe, 'avg_weekly_sell_qta')
    sells_dataframe = merge_by_day(avg_monthly_sells_data, sells_dataframe, 'avg_monthly_sell_qta')
    sells_dataframe = merge_by_day(avg_quarter_sells_data, sells_dataframe, 'avg_quarter_sell_qta')
    sells_dataframe = merge_by_day(avg_quad_sells_data, sells_dataframe, 'avg_quad_sell_qta')
    sells_dataframe['day_of_week'] = sells_dataframe.index.day_of_week

    return sells_dataframe, weekly_sells_data.index, monthly_sells_data.index, quarter_sells_data.index, quad_sells_data.index

def insights(buy_data, sells_data, start_date_fill, work_days=6):
    days_lag = orders_days_of_lag(buy_data.index)
    resampled_days_lag = resample(start_date_fill, days_lag, 'days_lag')

    sells_dataframe, weekly_sells_data_index, monthly_sells_data_index, quarter_sells_data_index, quad_sells_data_index = \
        sales_resample(sells_data, start_date_fill, work_days=work_days)

    resampled_buy_data = resample(
        start_date_fill,
        buy_data,
        ['buy_qta', 'buy_delivered', 'buy_cost'],
        aggregator=lambda d: d.agg({'buy_qta':'sum', 'buy_delivered':'sum', 'buy_cost':'mean'}))


    buy_sell_data = pd.merge(resampled_buy_data, sells_dataframe, how='outer', left_index=True, right_index=True).fillna(0)
    buy_sell_data['stock'] = (buy_sell_data.buy_delivered - buy_sell_data.sell_qta).cumsum()
    buy_sell_data = merge_by_day(resampled_days_lag, buy_sell_data, 'days_lag', 'days_lag')

    return buy_sell_data, weekly_sells_data_index, monthly_sells_data_index, quarter_sells_data_index, quad_sells_data_index

def median_quantile(sells_data: pd.DataFrame, replace = True, lower = 0.25, higher = 0.75):
    Q1 = sells_data.quantile(lower)
    Q3 = sells_data.quantile(higher)
    IQR = Q3 - Q1

    sells_data_quantile = sells_data[~((sells_data < (Q1 - 1.5 * IQR)) | (sells_data > (Q3 + 1.5 * IQR))).any(axis=1)]

    if replace:
        diff_idx = sells_data.index.difference(sells_data_quantile.index)
        remainder = pd.DataFrame([sells_data.quantile(0.5)] * diff_idx.size, index=diff_idx)
        sells_data_quantile = pd.concat([sells_data_quantile, remainder], axis=0).sort_index()

    return sells_data_quantile

def quantile(xd: pd.DataFrame, lower = 0.25, higher = 0.75):
    Q1 = xd.quantile(lower)
    Q3 = xd.quantile(higher)
    IQR = Q3 - Q1
    min = Q1 - 1.5 * IQR
    max = Q3 + 1.5 * IQR
    outliers = xd[((xd < min) | (xd > max)).any(axis=1)]
    median = xd.quantile(0.5)
    remapped = outliers.loc[:, outliers.columns[0]].map(lambda x: max[0] if x > median[0] else min[0])

    xd_quantile = xd.copy()
    xd_quantile.update(remapped)

    return xd_quantile

def nn_train_dataset(buy_sell_data, weekly_index, start_date='2016-01-01'):
    avg_weekly_sells_data = buy_sell_data.loc[weekly_index].avg_weekly_sell_qta
    sells = avg_weekly_sells_data[start_date:]

    def previous_period(series, d, days_to_use, back_of_days=365):
        start_date = d - pd.to_timedelta(back_of_days, unit='d')
        end_date = start_date + pd.to_timedelta(days_to_use, unit='d')
        return series[start_date: end_date]

    def subtract(series, d, days):
        return series[d - pd.to_timedelta(days, unit='d'): d]

    def sample(data_set, date, week_avg, next_week_avg):
        return date, week_avg, \
            subtract(data_set, date, 30).mean(), \
            subtract(data_set, date, 90).mean(), \
            previous_period(data_set, date, 30).mean(), \
            previous_period(data_set, date, 90).mean(), \
            next_week_avg

    return np.array([sample(avg_weekly_sells_data, d, w, w1) for d, w, w1 in zip(sells.index, sells, sells[1:])])

def nn_train_dataframe(buy_sell_data, weekly_index, start_date='2016-01-01'):
    train_data = nn_train_dataset(buy_sell_data, weekly_index)
    columns = ['week', 'month', 'quarter', 'prev_month', 'prev_quarter', 'actual']
    train_df = pd.DataFrame(
        # np.concatenate((train_data[:, 1:4], train_data[:, 6].reshape(train_data.shape[0], 1)), axis=1),
        train_data[:, 1:],
        columns=columns)
    train_df.set_index(pd.DatetimeIndex(train_data[:, 0]), inplace=True)

    return train_df

def nn_train_df_from_minsan(minsan, data_dir, initial_stock = 0):
    sells_data = read_sell(minsan, data_dir)
    buy_data = read_buy(minsan, data_dir, initial_stock = initial_stock)

    import datetime
    start_date_fill = datetime.date(2015, 1, 1)

    buy_sell_data, weekly_index, monthly_index, quarter_index = \
        insights(buy_data, sells_data, start_date_fill)

    return nn_train_dataframe(buy_sell_data, weekly_index)

def xslice(arr, slices):
    if isinstance(slices, tuple):
        return sum((arr[s] if isinstance(s, slice) else [arr[s]] for s in slices), [])
    elif isinstance(slices, slice):
        return arr[slices]
    else:
        return [arr[slices]]

def fft_denoiser(x, n_components, to_real=True):
    """Fast fourier transform denoiser.

    Denoises data using the fast fourier transform.

    Parameters
    ----------
    x : numpy.array
        The data to denoise.
    n_components : int
        The value above which the coefficients will be kept.
    to_real : bool, optional, default: True
        Whether to remove the complex part (True) or not (False)

    Returns
    -------
    clean_data : numpy.array
        The denoised data.

    References
    ----------
    .. [1] Steve Brunton - Denoising Data with FFT[Python]
       https://www.youtube.com/watch?v=s2K1JfNR7Sc&ab_channel=SteveBrunton

    """
    n = len(x)

    # compute the fft
    fft = np.fft.fft(x, n)

    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n
    PSD = PSD

    # keep high frequencies
    _mask = np.zeros(len(PSD))
    # _mask[50] = 1.0
    # _mask[71] = 1.0
    # _mask[370 - 71] = 1.0
    # _mask[370 - 50] = 1.0

    psdDf = pd.DataFrame(PSD)
    quantile = psdDf.quantile(n_components)
    _mask = PSD > quantile[0]
    fft = _mask * fft

    # inverse fourier transform
    clean_data = np.fft.ifft(fft)

    if to_real:
        clean_data = clean_data.real
    return clean_data, pd.Series(PSD * _mask), pd.Series(PSD)

def fft_denoiser_series(x, n_components, index):
    denoised, mask, psd = fft_denoiser(x, n_components)
    return pd.Series(denoised, index = index), mask, psd

