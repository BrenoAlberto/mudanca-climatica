import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LinReg


plt.style.use('ggplot')

df = pd.read_csv('GlobalTemperatures.csv')

# Considera apenas as colunas de index, dt e LandAverageTemperature(Temperatura Terrestre Média)
df = df.ix[:, :2]

# Converte as datas str para datetime
times = pd.DatetimeIndex(df['dt'])

# Usa observações anteriores válidas para preencher os gaps NaN
df['LandAverageTemperature'] = df['LandAverageTemperature'].fillna(method='ffill')

# Agrupa as medias por ano
grouped = df.groupby([times.year]).mean()

# Monta o graph
plt.figure(figsize = (15, 5))
plt.plot(grouped['LandAverageTemperature'])
plt.title("Temperatura Terrestre Média 1750-2015")
plt.xlabel("Ano")
plt.ylabel("Temperatura Terrestre Média")
plt.show()

# Model || Regressão Linear
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



