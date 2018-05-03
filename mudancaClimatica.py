import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LinReg


plt.style.use('ggplot')

df = pd.read_csv('GlobalTemperatures.csv')

# Consider just the cols of index, dt and LandAverageTemperature
df = df.ix[:, :2]

# Convert the dates str to datetime
times = pd.DatetimeIndex(df['dt'])

# Use previous valid observations to fill the NaN gaps
df['LandAverageTemperature'] = df['LandAverageTemperature'].fillna(method='ffill')

# Group the means by year
grouped = df.groupby([times.year]).mean()

# Graph
plt.figure(figsize = (15, 5))
plt.plot(grouped['LandAverageTemperature'])
plt.title("Temperatura Terrestre Média 1750-2015")
plt.xlabel("Ano")
plt.ylabel("Temperatura Terrestre Média")
plt.show()

# Model || Linear Regression
x = grouped.index.values.reshape(-1, 1)
y = grouped['LandAverageTemperature'].values

reg = LinReg()
reg.fit(x, y)
y_pred = reg.predict(x)
print("Precisão: " + str(reg.score(x, y)))

plt.figure(figsize = (15, 5))
plt.title("Regressão Linear")
plt.scatter(x = x, y = y_pred)
plt.scatter(x = x, y = y, c = "r")
plt.show()


print("Predict para temperatura em 2050: " + str(reg.predict(2050)))



