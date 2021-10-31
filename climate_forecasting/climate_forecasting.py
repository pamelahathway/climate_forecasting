import os
import time
from typing import List, Tuple
import itertools as it

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import sweetviz as sv

# todo: add pandas configuration


# parameters
data_path = "/Users/ph/Documents/ds_projects/climate_change/data"


# 0) load data
def load_csv_file(filename: str):
    print(f"loading {filename}")
    df = pd.read_csv(os.path.join(data_path, f"{filename}.csv"))
    df["dt"] = pd.to_datetime(df["dt"])
    return df


glob_temp = load_csv_file("GlobalTemperatures")
# glob_temp_city = load_csv_file("GlobalLandTemperaturesByCity")
# glob_temp_country = load_csv_file("GlobalLandTemperaturesByCountry")
# glob_temp_major_city = load_csv_file("GlobalLandTemperaturesByMajorCity")
# glob_temp_state = load_csv_file("GlobalLandTemperaturesByState")


glob_temp = glob_temp.set_index("dt")

# plot average land and ocean temperatures
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(glob_temp.index, glob_temp["LandAverageTemperature"], '.-', linewidth=0.5)
plt.show()


# resample to yearly or monthly data
avg_temp_per_year = glob_temp.resample(rule="Y").mean()
avg_temp_per_month = glob_temp.resample(rule="M").mean()


fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=False)

# plot avg temperature per year
ax[0].fill_between(avg_temp_per_year.index, avg_temp_per_year["LandAverageTemperature"]-avg_temp_per_year["LandAverageTemperatureUncertainty"], avg_temp_per_year["LandAverageTemperature"]+avg_temp_per_year["LandAverageTemperatureUncertainty"], alpha=0.2, label="uncertainty")
ax[0].plot(avg_temp_per_year.index, avg_temp_per_year["LandAverageTemperature"], linestyle='-', linewidth=2, label="temperature")
ax[0].set_title("Average temperature per year")
# plot avg temperature per month
ax[1].fill_between(avg_temp_per_month.index, avg_temp_per_month["LandAverageTemperature"]-avg_temp_per_month["LandAverageTemperatureUncertainty"], avg_temp_per_month["LandAverageTemperature"]+avg_temp_per_month["LandAverageTemperatureUncertainty"], alpha=0.2, label="uncertainty")
ax[1].plot(avg_temp_per_month.index, avg_temp_per_month["LandAverageTemperature"], linestyle='-', linewidth=0.5, label="temperature")
ax[1].set_xlabel("year")
ax[1].set_title("Average temperature per month")

for ax_idx in range(len(ax)):
    ax[ax_idx].set_ylabel("temperature [C]")
    ax[ax_idx].grid(True)
    ax[ax_idx].legend()
plt.tight_layout()
plt.show()

# pivot to get month_vs_year
month_vs_year = avg_temp_per_month.pivot_table(values='LandAverageTemperature',index=avg_temp_per_month.index.month,columns=avg_temp_per_month.index.year)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
month_vs_year.plot(legend=False, linewidth=0.1, color='k', ax=ax[0])
sns.heatmap(month_vs_year, ax=ax[1])
plt.tight_layout()
plt.show()

# conclusions from this:
# yearly data:
# - seasonal and random fluctuations change as time goes on (more fluctuations at earlier times)
#     therefore it might be wise to only use part of the time series when the noise is relatively stationary
#     e.g. only use data from ca 1840 onwards

# monthly data:
# - seasonal and random fluctuations do not change a lot as time goes on
# - very strong seasonal effect --> should probably use SARIMA or something similar
# - min and max per month looks stable - are the spring and autumn getting warmer?


avg_temp_per_year = avg_temp_per_year[avg_temp_per_year.index.year >= 1850]

# filter for relevant columns
# avg_temp_per_year = pd.DataFrame(avg_temp_per_year["LandAverageTemperature"])

# test for non-stationarity
# ADF test determines whether the change in Y can be explained by a lagged value
# (e.g., a value at a previous time point Y [t-1] ) and by a linear trend.
# If there is a linear trend but the lagged value cannot explain the change in Y over time,
# then our data will be deemed non-stationary.


def run_adf_test(df: pd.DataFrame, column: str):
    result = adfuller(df[column], autolag='AIC')
    p_value = result[1]
    print(f"Testing {column}")
    print(f"    Null Hypothesis for augmented Dickie Fuller: the time series is non-stationary")

    print(f"    p = {p_value:.2f}")
    if p_value > 0.05:
        print("    --> we cannot reject the Null Hypothesis -> the time series is non-stationary")
        print("        we will have to differentiate the time series further")
        time_series_is_stationary = False
    else:
        print("        --> we can reject the Null Hypothesis -> the time series is stationary.")
        time_series_is_stationary = True
    return time_series_is_stationary


# check stationarity
time_series_is_stationary = run_adf_test(avg_temp_per_year, "LandAverageTemperature")
avg_temp_per_year_diff = avg_temp_per_year.diff()
time_series_is_stationary = run_adf_test(avg_temp_per_year_diff.dropna(), "LandAverageTemperature")

# --> looks to be a random walk with drift



# check autocorrelation
"""
AUTOCORRELATION is just like a correlation, except that, rather than correlating two completely different variables, 
it’s correlating a variable at time t and that same variable at time t-k

A partial autocorrelation is basically the same thing, except that it removes the effect of shorter autocorrelation lags 
when calculating the correlation at longer lags. To be more precise, the partial correlation at lag k is the 
autocorrelation between Yt and Yt-k that is NOT accounted for by the autocorrelations from the 1st to the (k-1)st lags.

Autocorrelation assumes the variables to be distributed normally
"""

# plot ACF and PACF
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
plot_acf(avg_temp_per_year["LandAverageTemperature"], lags=50, ax=axes[0])
plot_pacf(avg_temp_per_year["LandAverageTemperature"], lags=50, ax=axes[1])
plt.xlabel("lag")
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
plot_acf(avg_temp_per_year["LandAverageTemperature"].diff().dropna(), lags=50, ax=axes[0])
plot_pacf(avg_temp_per_year["LandAverageTemperature"].diff().dropna(), lags=50, ax=axes[1])
plt.xlabel("lag")
plt.tight_layout()
plt.show()


# ARIMA parameters
# p = AR (autoregression) order:
# d = derivative: 1 (number of times the series needs to be "differenced" to become stationary
# q = MA (moving average) order:

# parameter search
d_values_to_try = range(1, 3)
p_values_to_try = range(4)
q_values_to_try = range(4)
p_q_combinations = list(it.product(p_values_to_try, d_values_to_try, q_values_to_try))

parameter_results = []
for p, d, q in p_q_combinations:
    mod = ARIMA(avg_temp_per_year["LandAverageTemperature"], order=(p, d, q))
    fitted_model = mod.fit()
    print(f"{p=} {d=} {q=}")
    # print(fitted_model.mle_retvals)
    parameter_results.append([p, d, q, fitted_model.aic, fitted_model.bic, fitted_model.aicc, fitted_model.mle_retvals["converged"]])

parameter_results = pd.DataFrame(data=parameter_results, columns=["p", "d", "q", "aic", "bic", "aicc", "converged"])

results_d_is_1 = parameter_results[parameter_results["d"] == 1]
results_d_is_2 = parameter_results[parameter_results["d"] == 2]

fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
sns.heatmap(results_d_is_1.pivot(values="aic", columns="q", index="p"), ax=ax[0, 0])
sns.heatmap(results_d_is_1.pivot(values="aicc", columns="q", index="p"), ax=ax[0, 1])
sns.heatmap(results_d_is_1.pivot(values="bic", columns="q", index="p"), ax=ax[0, 2])
sns.heatmap(results_d_is_2.pivot(values="aic", columns="q", index="p"), ax=ax[1, 0])
sns.heatmap(results_d_is_2.pivot(values="aicc", columns="q", index="p"), ax=ax[1, 1])
sns.heatmap(results_d_is_2.pivot(values="bic", columns="q", index="p"), ax=ax[1, 2])
ax[0, 0].set_title("AIC")
ax[0, 1].set_title("AICC")
ax[0, 2].set_title("BIC")
ax[0, 0].set_ylabel("d=1")
ax[1, 0].set_ylabel("d=2")
plt.tight_layout()
plt.show()

NUM_YEARS_FOR_TRAINING = 100
YEARS_TO_FORECAST = 10


def run_backtests(series: pd.DataFrame, order: Tuple = (1, 1, 1)) -> pd.DataFrame:
    # create backtest training and test sets

    num_backtests = (series.shape[0] - 100) // YEARS_TO_FORECAST
    first_year_idx = series.shape[0] - 100 - YEARS_TO_FORECAST*num_backtests

    result_collection = []

    fig, ax = plt.subplots(1)
    ax.plot(series, c="g")

    for start_idx in range(first_year_idx, series.shape[0]-NUM_YEARS_FOR_TRAINING, YEARS_TO_FORECAST):
        end_idx = start_idx + NUM_YEARS_FOR_TRAINING
        start_year = series.index[start_idx].year
        start_prediction_year = series.index[end_idx].year
        mod = ARIMA(series.iloc[start_idx:end_idx], order=order)
        fitted_model = mod.fit()

        # Get forecast
        forecast_yearly = fitted_model.get_forecast(steps=YEARS_TO_FORECAST).predicted_mean
        # forecast_yearly.index = forecast_yearly.index.rename(avg_temp_per_year.index.name)

        true_temperatures = series[forecast_yearly.index]

        mse = mean_squared_error(forecast_yearly, true_temperatures)
        mase = mean_absolute_error(forecast_yearly, true_temperatures)
        mape = mean_absolute_percentage_error(forecast_yearly, true_temperatures)

        # todo: plot all
        print(p, d, q, start_year, start_prediction_year, round(mse, 3), round(mase, 3), round(mape, 3))

        ax.plot(forecast_yearly, c="red")
        ax.set_title(f"Forecast for years starting {start_prediction_year}\n{p=} {d=} {q=}")

        result_collection.append([p, d, q, start_year, start_prediction_year, mse, mase, mape])

    result_collection = pd.DataFrame(data=result_collection, columns=["p", "d", "q", "start_year", "start_prediction_year", "mse", "mase", "mape"])

    return result_collection, fig


d_values_to_try = range(1, 3)
p_values_to_try = range(4)
q_values_to_try = range(4)
p_q_combinations = list(it.product(p_values_to_try, d_values_to_try, q_values_to_try))

backtest_results = []
for p, d, q in p_q_combinations:
    print(f"{p=} {d=} {q=}")
    backtest_results_temp, fig = run_backtests(series=avg_temp_per_year["LandAverageTemperature"], order=(p, d, q))
    backtest_results.append(backtest_results_temp)

backtest_results = pd.concat(backtest_results)

backtest_results[backtest_results["d"] == 1].groupby(by=["start_year"]).mean()
backtest_results.groupby(by=["start_year"]).mean()

mean_performance_per_order = backtest_results.groupby(by=["p", "d", "q"]).mean()
ax = mean_performance_per_order[["mse", "mase", "mape"]].sort_values(by="mse").plot(figsize=(10, 4))
ax.set_xticks(range(mean_performance_per_order.shape[0]))
ax.set_xticklabels(mean_performance_per_order.index, rotation=45, ha="right")
ax.set_xlabel("average performance across all backtests")
ax.set_title("performance of ARIMA model (sorted)")
plt.tight_layout()


backtest_results.set_index(["p", "d", "q"])


test_set_perc = 0.1
years_to_forecast = int(avg_temp_per_year.shape[0]*test_set_perc)
idx_train_test_split = avg_temp_per_year.shape[0] - years_to_forecast

avg_temp_per_year_train = avg_temp_per_year.iloc[:idx_train_test_split]
avg_temp_per_year_test = avg_temp_per_year.iloc[idx_train_test_split:]




plt.rcParams['figure.figsize'] = [10, 6]

# Forecast temperatures using an ARIMA model
mod = ARIMA(avg_temp_per_year_train["LandAverageTemperature"], order=(1, 2, 1))
fitted_model = mod.fit()
print(round(fitted_model.aic, 2))

# Get forecast
forecast_yearly = fitted_model.get_forecast(steps=years_to_forecast)
forecast_yearly_ci = forecast_yearly.conf_int()

plt.figure()
plt.plot(forecast_yearly.predicted_mean, color='r', label="prediction")
plt.fill_between(forecast_yearly.row_labels,
                forecast_yearly_ci.iloc[:, 0],
                forecast_yearly_ci.iloc[:, 1], color='r', alpha=.2, label="confidence interval")
plt.plot(avg_temp_per_year["LandAverageTemperature"], color='k', label="observation")
plt.plot(fitted_model.predict(), color='orange', label="prediction")
plt.xlabel('Year')
plt.ylabel('average yearly temperature')
plt.title('Prediction')
plt.legend()
plt.show()


# line plot of residuals
residuals = pd.DataFrame(fitted_model.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())

# residuals mean of 0 suggests that there is low bias


# evaluate model performance

mse = mean_squared_error(forecast_yearly.predicted_mean, avg_temp_per_year_test["LandAverageTemperature"])
mase = mean_absolute_error(forecast_yearly.predicted_mean, avg_temp_per_year_test["LandAverageTemperature"])
mape = mean_absolute_percentage_error(forecast_yearly.predicted_mean, avg_temp_per_year_test["LandAverageTemperature"])


# train Seasonal ARIMA model
mod = SARIMAX(avg_temp_per_month["LandAverageTemperature"],
                                order=(2, 1, 2),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
fitted_model = mod.fit()
print(round(fitted_model.aic,2))

# Get forecast
years_to_forecast = 10
forecast_monthly = fitted_model.get_forecast(steps=12*years_to_forecast)
forecast_monthly_ci = forecast_monthly.conf_int()

plt.figure()
plt.plot(forecast_monthly.predicted_mean, color='orange', label="prediction")
plt.fill_between(forecast_monthly.row_labels,
                forecast_monthly_ci.iloc[:, 0],
                forecast_monthly_ci.iloc[:, 1], color='orange', alpha=.2, label="confidence interval")
plt.plot(avg_temp_per_month["LandAverageTemperature"][-12*years_to_forecast:], color='green', label="observation")
plt.xlabel('Year')
plt.ylabel('average monthly temperature')
plt.title('Prediction')
plt.legend()
plt.show()


print(fitted_model.summary().tables[1])
fitted_model.plot_diagnostics(figsize=(10, 7))
plt.tight_layout()
plt.show()
# evaluate goodness of fit!



# additions if time

# use SARIMAX to add exogenous variables e.g. CO2, other unrelated stuff

