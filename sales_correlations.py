#%%
initial_stock=0
minsan = '004763114' # ASPIRINA C*10CPR EFF 400+240MG
# minsan = 'A0420030101' # BD Category (fever and cold) (aspirina correlates to this)
# minsan = 'X0144010101' # BD Category (medicine under prescription)
# minsan = '024840074' # CARDIOASPIRIN*30CPR GAST 100MG it presents regular purchase pattern
# minsan = '011782012' # medicine SINTROM*20CPR 4MG
# minsan = '023853031' # CEFAZOLINA TEVA*IM 1F 1G+F 4ML
# minsan = '031981311' # PANTORC*14CPR GASTR 40MG
# minsan = '036635011' # DIBASE*OS GTT 10ML 10000UI/ML
# minsan = '024402051' # EUTIROX*50CPR 75MCG
# minsan = '016366027' # COUMADIN*30CPR 5MG
data_dir='data'

from scipy.misc import derivative
import common.data as data
import pandas as pd
import numpy as np

sales_data = data.read_sell(minsan, data_dir)
work_days = 7

import datetime
start_date_fill = datetime.date(2001, 1, 1)

buy_sell_data, weekly_index, monthly_index, quarter_index, quad_index = \
    data.sales_resample(sales_data, start_date_fill, work_days = work_days)

avg_weekly_sales_data = buy_sell_data.loc[weekly_index].avg_weekly_sell_qta
avg_sales_data = pd.DataFrame(avg_weekly_sales_data)
avg_sales_data.columns.values[0] = 'qta'

sales_data = buy_sell_data[['sell_qta']]
avg_sales_data_quantile = data.quantile(avg_sales_data)

import seaborn as sns
import matplotlib.pyplot as plt

all_sales = pd.concat( \
    [pd.DataFrame({'qta': avg_sales_data.qta, 'type': 'noisy'}), \
     pd.DataFrame({'qta': avg_sales_data_quantile.qta, 'type': 'quant'})]
)

grid = sns.FacetGrid(all_sales, col='type', height=4, aspect=1.5)
grid.map(sns.boxplot, 'qta', order=['noisy', 'quantile'])
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
dayly_sales_data = data.leave_work_days(buy_sell_data, work_days).sell_qta
weekly_sales_data = buy_sell_data.loc[weekly_index].avg_weekly_sell_qta
monthly_sales_data = buy_sell_data.loc[monthly_index].avg_monthly_sell_qta
quarter_sales_data = buy_sell_data.loc[quarter_index].avg_quarter_sell_qta
quad_sales_data = buy_sell_data.loc[quad_index].avg_quad_sell_qta

dayly_sales_data_quantile = data.quantile(pd.DataFrame(dayly_sales_data)).sell_qta
weekly_sales_data_quantile = data.quantile(pd.DataFrame(weekly_sales_data)).avg_weekly_sell_qta
monthly_sales_data_quantile = data.quantile(pd.DataFrame(monthly_sales_data)).avg_monthly_sell_qta
quarter_sales_data_quantile = data.quantile(pd.DataFrame(quarter_sales_data)).avg_quarter_sell_qta
quad_sales_data_quantile = data.quantile(pd.DataFrame(quad_sales_data)).avg_quad_sell_qta

def my_plot_acf(noisy, quantiled, lags, title):
    _, ax = plt.subplots(1,2,figsize=(15,6))
    # plot_acf(x=noisy[1:-1], lags=lags, title=f'{title}', ax=ax[0])
    plot_acf(x=noisy[1:-1], lags=lags, title=f'{title} noisy', ax=ax[0])
    plot_pacf(x=noisy[1:-1], lags=lags, title=f'{title} partial', ax=ax[1])

my_plot_acf(dayly_sales_data, dayly_sales_data_quantile, lags=14, title='daily sales')
my_plot_acf(weekly_sales_data, weekly_sales_data_quantile, lags = 53, title='weekly sales')
my_plot_acf(monthly_sales_data, monthly_sales_data_quantile, lags=13, title='monthly sales')
my_plot_acf(quarter_sales_data, quarter_sales_data_quantile, lags=5, title='quarter sales')
my_plot_acf(quad_sales_data, quad_sales_data_quantile, lags=5, title='quad sales')

sales = monthly_sales_data
sales_quantile = monthly_sales_data_quantile

weekly_sales_data_derivative = pd.Series(np.gradient(weekly_sales_data.values), index=weekly_index)

data.scatter_lines([weekly_sales_data, weekly_sales_data_quantile, weekly_sales_data_derivative])
data.scatter_lines([monthly_sales_data, monthly_sales_data_quantile])
data.scatter_lines([quarter_sales_data, quarter_sales_data_quantile])
data.scatter_lines([quad_sales_data, quad_sales_data_quantile])

import plotly.express as px
day_week_sales = buy_sell_data.groupby("day_of_week")['sell_qta'].mean()
fig = px.line(day_week_sales)
fig.show()
weekly_sales_data_1, mask_1, _ = data.fft_denoiser_series(weekly_sales_data_quantile, 0.5, weekly_sales_data.index)
weekly_sales_data_2, mask_2, _ = data.fft_denoiser_series(weekly_sales_data_quantile, 0.9, weekly_sales_data.index)
weekly_sales_data_3, mask_3, _ = data.fft_denoiser_series(weekly_sales_data_quantile, 0.95, weekly_sales_data.index)
weekly_sales_data_4, mask_4, _ = data.fft_denoiser_series(weekly_sales_data_quantile, 0.99, weekly_sales_data.index)
data.scatter_lines([
    weekly_sales_data_quantile,
    weekly_sales_data_1,
    weekly_sales_data_2,
    weekly_sales_data_3,
    weekly_sales_data_4
])

data.scatter_lines([
    abs(mask_1[1:]),
    abs(mask_2[1:]),
    abs(mask_3[1:]),
    abs(mask_4[1:])
])
