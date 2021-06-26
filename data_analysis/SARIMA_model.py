import warnings
import matplotlib as matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

import warnings
warnings.filterwarnings("ignore")
import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib
import scipy
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
warnings.filterwarnings("ignore")

df = pd.read_excel('SARIMA_data.xlsx', sheet_name='daily_monday', index_col='Time', parse_dates=True)

y = df
y.head()
y.plot(figsize=(19, 4))

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter for SARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

res = []
smallest_aics = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            #print('SARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))

            r = ('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
            print(r)
            res.append(r)
            if smallest_aics == None or results.aic < smallest_aics[0]:
                smallest_aics = results.aic, param, param_seasonal

        except:
            continue

print("----------------------\n\n\n")
for aic in res:
    print(aic)
print("smallest aics:", smallest_aics)

import datetime
train_end = datetime.datetime(2020,8,27)
test_end = datetime.datetime(2020,9,20)
train_data = df[:train_end]
test_data = df[train_end + datetime.timedelta(days=1):test_end]

mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 24),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(18, 8))
#plt.show()

pred = results.get_prediction(start=pd.to_datetime('2020-08-4'), dynamic=False)
pred_ci = pred.conf_int()
ax = test_data['2020':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Occupied parking spaces')
axes = plt.gca()
axes.set_ylim([0,80])
plt.legend()
plt.savefig("current.svg", format="svg")


pred_uc = results.get_forecast(steps=24) # meaning 24 hours
pred_ci = pred_uc.conf_int()
ax = test_data['2020':].plot(label='observed', figsize=(14, 4))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
pred_ci.iloc[:, 0],
pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Occupied Spaces')
axes = plt.gca()
axes.set_ylim([0,80])
plt.legend()
#plt.savefig("predict.svg", format="svg")
plt.show()



# accuarcy calculation
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    minmax = 0
    acf1 = acf(forecast-actual)[1]              # ACF1

    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1,
            'corr':corr, 'minmax':minmax})



# convert forecast and actual data to arrays for accuarcy calculation
forecast = pred_uc.predicted_mean # forecast data
forecast = np.array(forecast) # to normal array
print("forecast", forecast)
actual = test_data
actual = actual['Cars'].values.tolist() # to normal array

print("\n--------\n"
      "Key figures\n", forecast_accuracy(forecast, actual))
print(scipy.stats.linregress(forecast, actual))



