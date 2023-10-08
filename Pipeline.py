# =============================================================================
# File name      : pipeline.py
# Author         : Sandra López
# Date created   : 2023-10-07
# Description    : reads train.csv, runs different time-series analysis tasks in sequence
#                  and gives back a csv with the prediction for next 12 steps.
# =============================================================================
# prerequisites :
#   - train.csv should be located in the same place as the pipeline
#   - detailed explanations of the analysis can be found in the jupyter notebook
# =============================================================================


# -------------------------------------------------------------------------
# 0. required modules 
# -------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import statsmodels.api as sm
import itertools

# ignore warnings for whole process
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# 1. data load and preprocessing
# -------------------------------------------------------------------------

df=pd.read_csv('train.csv')

# Give columns an appropriate name and drop empty registers
df.rename(columns = {'Unnamed: 0':'date'}, inplace = True)
df.dropna( inplace=True)

#Change data into time-series format
df['date'] = pd.to_datetime(df["date"],dayfirst=True)
df = df.set_index('date')
df=df.asfreq(freq='MS')

    # -------------------------------------------------------------------------
    # 1.1. Train-test split
    # -------------------------------------------------------------------------

#We take last 20 dates as test sample (between 20% and 30% of sample size)
df_train=df[:-20]
df_test=df[-20:]

#Plotting the split
fig, ax=plt.subplots(figsize=(7, 3))
df_train['y'].plot(ax=ax, label='train')
df_test['y'].plot(ax=ax, label='test')
ax.legend()
ax.set_title('Train-test split')

# -------------------------------------------------------------------------
# 2. EDA
# -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 2.1. Stationary behaviour
    # -------------------------------------------------------------------------

# Stationary Test (Dickey-Fuller and KPSS)
# ==============================================================================

# We test the original series together with 3 next diffs
df_diff_1 = df_train.diff().dropna()
df_diff_2 = df_diff_1.diff().dropna()
df_diff_3 = df_diff_2.diff().dropna()
df_diff_4 = df_diff_3.diff().dropna()

print('Stationary Test original series')
print('-------------------------------------')
adfuller_result = adfuller(df_train)
kpss_result = kpss(df_train)
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

print('\nStationary Test diff series (order=1)')
print('--------------------------------------------------')
adfuller_result = adfuller(df_diff_1)
kpss_result = kpss(df_diff_1)
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

print('\nStationary Test diff series (order=2)')
print('--------------------------------------------------')
adfuller_result = adfuller(df_diff_2)
kpss_result = kpss(df_diff_2)
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

print('\nStationary Test diff series (order=3)')
print('--------------------------------------------------')
adfuller_result = adfuller(df_diff_3)
kpss_result = kpss(df_diff_3)
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

print('\nStationary Test diff series (order=4)')
print('--------------------------------------------------')
adfuller_result = adfuller(df_diff_4)
kpss_result = kpss(df_diff_4)
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')


# Based on test results we could get an idea of possible values for parameter d of ARIMA model

    # -------------------------------------------------------------------------
    # 2.2. Autocorrelation
    # -------------------------------------------------------------------------

#  Autocorrelation plot for original and diff series (order 1)
# ==============================================================================
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 4), sharex=True)
plot_acf(df_train, ax=axs[0], lags=10, alpha=0.05)
axs[0].set_title('Autocorrelation plot for original series')
plot_acf(df_diff_1, ax=axs[1], lags=10, alpha=0.05)
axs[1].set_title('Autocorrelation plot for diff series (order=1)')

# Based on test results we could get an idea of possible values for parameter q of ARIMA model

# Partial autocorrelation plot for original and diff series (order 1)
# ==============================================================================
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 3), sharex=True)
plot_pacf(df_train, ax=axs[0], lags=10, alpha=0.05)
axs[0].set_title('Partial autocorrelation plot for original series')
plot_pacf(df_diff_1, ax=axs[1], lags=10, alpha=0.05)
axs[1].set_title('Partial autocorrelation plot for diff series (order=1)')
plt.tight_layout()

# Based on test results we could get an idea of possible values for parameter p of ARIMA model

    # -------------------------------------------------------------------------
    # 2.3. Series decomposition
    # -------------------------------------------------------------------------
    
    # Series decomposition of original and diff series
# ==============================================================================
res_decompose = seasonal_decompose(df_train, model='additive', extrapolate_trend='freq')
res_descompose_diff_2 = seasonal_decompose(df_diff_1, model='additive', extrapolate_trend='freq')

fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9, 6), sharex=True)
res_decompose.observed.plot(ax=axs[0, 0])
axs[0, 0].set_title('Original series')
res_decompose.trend.plot(ax=axs[1, 0])
axs[1, 0].set_title('Trend')
res_decompose.seasonal.plot(ax=axs[2, 0])
axs[2, 0].set_title('Seasonality')
res_decompose.resid.plot(ax=axs[3, 0])
axs[3, 0].set_title('Residuals')
res_descompose_diff_2.observed.plot(ax=axs[0, 1])
axs[0, 1].set_title('Diff series (order=1)')
res_descompose_diff_2.trend.plot(ax=axs[1, 1])
axs[1, 1].set_title('Trend')
res_descompose_diff_2.seasonal.plot(ax=axs[2, 1])
axs[2, 1].set_title('Seasonality')
res_descompose_diff_2.resid.plot(ax=axs[3, 1])
axs[3, 1].set_title('Residuals')
fig.suptitle('Series decomposition of original and diff series', fontsize=14)
fig.tight_layout()

# We use this plots to get an idea of seasonality, among others

    # -------------------------------------------------------------------------
    # 2.4. Doublecheck stationarity after diff and removing seasonality effect
    # -------------------------------------------------------------------------

# 1st order diff combined with seasonality diff
# ==============================================================================
datos_diff_1_12 = df_train.diff().diff(12).dropna()

adfuller_result = adfuller(datos_diff_1_12)
print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
kpss_result = kpss(datos_diff_1_12)
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

# -------------------------------------------------------------------------
# 3. Modelling
# -------------------------------------------------------------------------

# Linear regression tends to be good at capturing trends and making forecasts, thus we'll use ARIMA-type models for this case, capturing as well autorregresive dependencies. 
#Since the series seems to have a seasonal component (see EDA results), we will use
# a SARIMAX-type regressor.

    # -------------------------------------------------------------------------
    # 3.1. Tuning SARIMAX hyperparameters - Define Grid from EDA results
    # -------------------------------------------------------------------------

# Create the grid
p = [0,1,3]
d=[1,2] 
q=range(0,3) 
pdq = list(itertools.product(p,d, q)) #all combinations for pdq
pdq = list(dict.fromkeys(pdq)) #drop duplicates
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p,d, q))]
seasonal_pdq = list(dict.fromkeys(seasonal_pdq)) #drop duplicates

    # -------------------------------------------------------------------------
    # 3.2. Tuning SARIMAX hyperparameters - Picking best-performing model over validation (test) set
    # -------------------------------------------------------------------------

rmse_list=[]

# loop through all parameter combinations and store results
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try: 
            modelo = SARIMAX(endog = df_train, order = param,seasonal_order = param_seasonal)
            modelo_res = modelo.fit(disp=0)
            predicciones_statsmodels = modelo_res.get_forecast(steps=len(df_test)).predicted_mean
            rmse = np.sqrt(np.square(np.subtract(predicciones_statsmodels,df_test['y'])).mean())
            rmse_list.append(rmse)
        #There might be parameter combinations that bring errors, thus we indicate the loop to ignore those cases
        except: 
            continue

# keep parameter combination that brings the lowest error
min_pos=np.array(rmse_list).argmin()
index_pdq=int(min_pos/len(seasonal_pdq))
index_season=min_pos%len(seasonal_pdq)
param_1=pdq[index_pdq]
param_2=seasonal_pdq[index_season]


    # -------------------------------------------------------------------------
    # 3.3. Evaluating model performance.
    # -------------------------------------------------------------------------

# best-performant model
modelo=SARIMAX(endog = df_train, order = param_1,seasonal_order = param_2)
modelo_res = modelo.fit(disp=0)

# evaluate model over validation set 
predicciones_statsmodels = modelo_res.get_forecast(steps=len(df_test)).predicted_mean

# Calculate the root mean squared error of best-performant model
rmse = np.sqrt(np.square(np.subtract(predicciones_statsmodels,df_test['y'])).mean())
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Error in test (%): {100*rmse/df_test.mean()}')

# -------------------------------------------------------------------------
# 4. Forecasting
# -------------------------------------------------------------------------

# evaluate model over validation set + 12 next steps
predicciones_statsmodels = modelo_res.get_forecast(steps=len(df_test)+12).predicted_mean
predicciones_statsmodels.name = 'y'

# Plot predictions
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 3))
df_train['y'].plot(ax=ax, label='train')
df_test['y'].plot(ax=ax, label='test')
predicciones_statsmodels.plot(ax=ax, label='prediction')
ax.set_title('Predictions')
ax.legend()

# Prepare sample with prediction for 12 next steps
predictions=predicciones_statsmodels[-12:]
predictions_df=pd.DataFrame(predictions)
predictions_df['date']=predictions_df.index
predictions_df=predictions_df.reset_index(drop=True)
predictions_df=predictions_df[['date','y']]
predictions_df['date']=predictions_df['date'].dt.strftime('%d.%m.%y')

# save sample
predictions_df.to_csv('test_set.csv',index=False)

# set warnings to default value
warnings.filterwarnings("default")