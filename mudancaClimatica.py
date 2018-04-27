import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


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





