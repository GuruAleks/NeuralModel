# %%
### Программа графического анализа данных ###

# импортируем и настраиваем параметры вывода графиков
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rcParams['axes.grid'] = True
mpl.rcParams['figure.figsize'] = (12, 6)

import pandas as pd
import numpy as np

# %%
# Загружеме нужный файл
version = '1.1.2'
filename = f'C:/Project/data/usdrub_{version}.csv'

stockdata = pd.read_csv(filename)
stockdata.set_index('DateTime', inplace=True)


# %%
# Выбираем данные нужного периода
stockdata_period = stockdata.query("'2020-11-02 08:00:00' <= DateTime < '2022-02-20'")
#print(stockdata_period.head(20))

lastpriceKF = stockdata_period['LastPriceKF'].to_numpy()
lastpriceDEV = stockdata_period['LastPriceDEV'].to_numpy()
lastpriceDIV = stockdata_period['LastPriceDIV'].to_numpy()

lastpricenorm = (lastpriceDIV - lastpriceDIV.mean())/lastpriceDIV.std()
print(lastpricenorm.min(), lastpricenorm.max())
print(lastpriceDIV.min(), lastpriceDIV.max())
# %%
#x_axes = stockdata_period.index.to_numpy()
#plt.subplot(2, 1, 1)
#plt.plot(x_axes, stockdata_period['LastPrice'].to_numpy(), 'r.')
#plt.plot(x_axes, lastpriceKF, color='blue')
#plt.plot(x_axes, lastpriceDEV, color='green')
#plt.subplot(2, 1, 2)
#plt.plot(x_axes, lastpriceDIV, color='black')
#plt.hist(lastprice_zeronorm, )
#plt.legend()
#plt.xlabel('Time-Step')
#plt.grid()
#plt.show()
sns.set_style('darkgrid')
sns_plot = sns.histplot(lastpricenorm)
fig = sns_plot.get_figure()
plt.show()
