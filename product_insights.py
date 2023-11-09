#%%
from tokenize import group
import pandas as pd
from common import data

initial_stock=0
minsan = '004763114' # ASPIRINA C*10CPR EFF 400+240MG
initial_stock=131
# minsan = '024840074' # CARDIOASPIRIN*30CPR GAST 100MG it presents regular purchase pattern
# initial_stock=220
# minsan = '011782012' # medicine SINTROM*20CPR 4MG
# minsan = '023853031' # CEFAZOLINA TEVA*IM 1F 1G+F 4ML (stock not affordable due to many missed sales)
# minsan = '031981311' # PANTORC*14CPR GASTR 40MG
# minsan = '036635011' # DIBASE*OS GTT 10ML 10000UI/ML
# minsan = '024402051' # EUTIROX*50CPR 75MCG
# initial_stock=16
# minsan = '016366027' # COUMADIN*30CPR 5MG
# initial_stock=8
data_dir='data'

def read_buy_data(minsan, data_dir):
    buy_data = pd.read_csv(f'{data_dir}\\orstd-{minsan}.csv', sep=';')
    buy_data = buy_data[['rec-orstd.orstd-key.orstd-tempo.orstd-data', 'rec-orstd.orstd-key.orstd-fornitore', 'rec-orstd.orstd-dati.orstd-qta']]
    buy_data.rename(columns={
        'rec-orstd.orstd-key.orstd-tempo.orstd-data' : 'date',
        'rec-orstd.orstd-key.orstd-fornitore': 'ws', # WholeSaler
        'rec-orstd.orstd-dati.orstd-qta': 'buy_qta'
    }, inplace=True, errors='raise')
    buy_data['ws'] = buy_data.ws.astype(str)
    buy_data['ws'] = buy_data.ws.str.rstrip()
    buy_data['ws'] = buy_data.ws.str.lstrip('0')
    buy_data = buy_data[buy_data['buy_qta'] > 0]
    buy_data['date'] = pd.to_datetime(buy_data.date, format='%Y%m%d')
    buy_data.sort_values(by='date', inplace=True)
    return buy_data

def read_sell_data(minsan, data_dir):
    sales_data = pd.read_csv(f'{data_dir}\\statava-{minsan}.csv', sep=';')
    sales_data = sales_data[['vb.vb-key.vb-data', 'vb.vb-dati.vb-qta']]
    sales_data.rename(columns={
        'vb.vb-key.vb-data' : 'date',
        'vb.vb-dati.vb-qta': 'sell_qta'
    }, inplace=True, errors='raise')
    sales_data['date'] = pd.to_datetime(sales_data.date, format='%Y%m%d')

    return sales_data

sales_data = data.read_sell(minsan, data_dir)
buy_data = buy_data = data.read_buy(minsan, data_dir, initial_stock = initial_stock)
wholesalers_group = buy_data.groupby('ws')
wholesalers_count = wholesalers_group.size()

# %% graph
import datetime
import plotly.graph_objects as go
start_date_fill = datetime.date(2015, 1, 1)

def resample(start_date_fill, data, column_name, time_frame = '1D'):
    data.set_index(pd.DatetimeIndex(data.date), inplace=True)
    data = data[data.index > str(start_date_fill)]
    start = pd.DataFrame([0], columns=[column_name], index=pd.DatetimeIndex([start_date_fill]))
    data = pd.concat([start, data])
    data = data.resample(time_frame).sum().pad()
    return data

resampled_sales_data = resample(start_date_fill, sales_data, 'sell_qta')
resampled_buy_data = resample(start_date_fill, buy_data, 'buy_qta')

buy_sell_data = pd.merge(resampled_buy_data, resampled_sales_data, how='outer', left_index=True, right_index=True).fillna(0)
buy_sell_data['stock'] = (buy_sell_data.buy_delivered - buy_sell_data.sell_qta).cumsum()
buy_sell_data.loc[str(start_date_fill), 'stock'] = buy_sell_data.loc[str(start_date_fill), 'stock'] + initial_stock
resampled_sales_data = resample(start_date_fill, sales_data, 'sell_qta', time_frame='1W')

fig = go.Figure(layout_title_text=f'{data_dir}: {minsan}')
fig = fig.add_trace(go.Line(x=resampled_sales_data.index, y = resampled_sales_data.sell_qta, name='sales'))
fig = fig.add_trace(go.Line(x=buy_sell_data.index, y = buy_sell_data.buy_qta, name='total buy'))
fig = fig.add_trace(go.Line(x=buy_sell_data.index, y = buy_sell_data.stock, name='stock'))
for g in wholesalers_group:
    data = resample(start_date_fill, g[1], 'buy_qta')
    fig.add_trace(go.Line(x=buy_sell_data.index, y = data.buy_qta, name=g[1].ws.values[0]))

monthly_sales_data = resample(start_date_fill, sales_data, 'sell_qta', time_frame='1M')
fig = fig.add_trace(go.Line(x=monthly_sales_data.index, y = monthly_sales_data.sell_qta, name='sales by month'))

fig.show()

# %% wholesalers stats
def days(timedelta):
    try:
        return timedelta.days
    except AttributeError:
        return timedelta

def wholesalers_days_stats(ws_group):
    ws = ws_group[1].reset_index(drop=True)
    orders_dates = pd.Series(ws['date'].unique())

    grouped_orders_dates = orders_dates.groupby(orders_dates.dt.year)
    stats_by_year = pd.DataFrame([stats(g[1]) for g in grouped_orders_dates]).fillna(pd.Timedelta(0))
    stats_by_year_mean = stats_by_year.mean()

    names = (ws['ws']).reset_index(drop=True)
    count, avg, std = stats(orders_dates)

    ws.set_index(pd.DatetimeIndex(ws.date), inplace=True)
    orders_volumes = ws.resample('1D').sum()
    orders_volumes = orders_volumes[orders_volumes['buy_qta'] > 0]['buy_qta']
    return (
        names[0],
        float(orders_volumes.sum()),
        float(orders_volumes.mean()),
        float(orders_volumes.median()),
        count,
        days(avg),
        days(std),
        days(stats_by_year_mean[0]),
        days(stats_by_year_mean[1]),
        days(stats_by_year_mean[2])
    )

def stats(orders_dates):
    coupled_dates = map(lambda p: p[1] - p[0], zip(orders_dates, orders_dates[1:]))
    coupled_dates_series = pd.Series(coupled_dates)
    return len(orders_dates), coupled_dates_series.mean(), coupled_dates_series.std()

wholesalers_stats = pd.DataFrame(
    [wholesalers_days_stats(g) for g in wholesalers_group],
    columns = ['ws', 'total volume', 'avg volume', 'med volume', 'day count', 'day avg', 'day std', 'day count / y', 'day avg / y', 'day std / y'])

wholesalers_stats