
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Question One
# load the data from file
df = pd.read_csv('JPYUSD.csv')
print(df.head())

# plot the series
plt.figure(figsize=(13,5))
plt.plot(df['Date'], df['Close'])
plt.xlabel("Date")
plt.ylabel("Ex Rate")

from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

print(auto_arima(df['Close'][:-10]))

# Fit the ARIMA model
sarima = ARIMA(df['Close'][:-10], order = (0, 1, 0),
               seasonal_order = (0, 0, 0, 12)).fit()
print(sarima.summary())

# Forecast 10-step ahead with ARIMA model
fseason = sarima.get_forecast(steps = 10).summary_frame()

plt.figure(figsize=(13, 5))

# Plot the outcome
plt.plot(df['Close'][:-10])

# Plot the seasonal results
plt.plot(fseason["mean"])
plt.fill_between(fseason.index, fseason["mean_ci_lower"],
                 fseason["mean_ci_upper"], color="grey", alpha=.1)
plt.xlabel("index")
plt.ylabel("JPYUSD")
plt.legend(("Actual", "Benchmark", "Seasonal forecast"))
plt.title("JPYUSD ARIMA Forecast")


# Question Two
t = np.array(np.arange(len(df)-10), ndmin=2).T
t2 = t ** 2
t3 = t ** 3
X_1 = t
X_2 = np.concatenate((t, t2), axis=1)
X_3 = np.concatenate((t, t2, t3), axis=1)
Y = df['Close'][:-10]

from sklearn.linear_model import LinearRegression

model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()

# sklearn automatically includes an intercept
model1.fit(X_1, Y)
model2.fit(X_2, Y)
model3.fit(X_3, Y)

fitted_1 = model1.predict(X_1)
fitted_2 = model2.predict(X_2)
fitted_3 = model3.predict(X_3)

# Plot
plt.figure(figsize=(13,5))
plt.plot(X_1, Y)
plt.plot(X_1, fitted_1)
plt.plot(X_1, fitted_2)
plt.plot(X_1, fitted_3)

plt.xlabel("index")
plt.ylabel("JPYUSD")
plt.legend(("Actual", "Linear", "Quadratic", "Cubic"))
plt.title("JPYUSD ARIMA Forecast")

X_1 = np.array(range(len(df) - 10,len(df)), ndmin=2).T
X_2 = np.concatenate((X_1, X_1 ** 2), axis=1)
X_3 = np.concatenate((X_1, X_1 ** 2, X_1 ** 3), axis=1)

# Forecast with Linear Model
fmodel_1 = model1.predict(X_1)
fmodel_2 = model2.predict(X_2)
fmodel_3 = model3.predict(X_3)

plt.figure(figsize=(13, 5))

# Plot the outcome
plt.plot(df['Close'][-30:])

# Plot the seasonal results
plt.plot(X_1,fmodel_1)
plt.plot(X_1,fmodel_2)
plt.plot(X_1,fmodel_3)
plt.xlabel("index")
plt.ylabel("JPYUSD")
plt.legend(("Actual", "Linear Forecast", "Quadratic Forecast", "Cubic Forecast"))
plt.title("JPYUSD Linear Forecast")

plt.plot(df['Close'][-30:])
plt.plot(X_1,fmodel_1)
plt.plot(X_1,fmodel_2)
plt.plot(X_1,fmodel_3)
plt.plot(X_1,fseason["mean"])
plt.xlabel("index")
plt.ylabel("JPYUSD")
plt.legend(("Actual", "Linear Forecast", "Quadratic Forecast", "Cubic Forecast", "ARIMA Forecast"))
plt.title("JPYUSD Forecast Results Comparison")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
hw_model = ExponentialSmoothing(Y, trend="add", seasonal="add",seasonal_periods=7).fit()
hw_prediction = hw_model.forecast(steps = 10)

plt.figure(figsize=(13,5))
plt.plot(df['Close'][:-10])
plt.plot(hw_model.level)

plt.figure(figsize=(13,5))
plt.plot(df['Close'][-30:])
plt.plot(X_1,hw_prediction)