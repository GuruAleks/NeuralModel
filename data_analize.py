# %%
### Программа графического анализа данных ###

# импортируем и настраиваем параметры вывода графиков
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 6)

import pandas as pd
import numpy as np

# %%
# Загружеме нужный файл
version = '1.0.2'
filename = f'C:/Project/data/usdrub_{version}.csv'

stockdata = pd.read_csv(filename)
stockdata.set_index('DateTime', inplace=True)
#stockdata.head()

# %%
# Выбираем данные нужного периода
stockdata_period = stockdata.query("'2021-02-08 10:00:00' <= DateTime < '2021-02-08 20:00:00'")
stockdata_period.head()

# %%
plt.plot(stockdata_period['VolumeSUM'], 'r.')
plt.plot(stockdata_period['VolumeKF'], color='blue')
#plt.legend()
plt.grid()
plt.show()


