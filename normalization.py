import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler


import matplotlib as mpl
import matplotlib.pyplot as plt


def positive_data(data:np.array):
    data_list = []
    for i in range(len(data)):
        if data[i] >= 0:
            data_list.append(data[i])
    pos_data = np.array(data_list)
    #print(f'Positive median: {np.median(pos_data)}')
    return pos_data

def negative_data(data:np.array):
    data_list = []
    for i in range(len(data)):
        if data[i] <= 0:
            data_list.append(data[i])
    neg_data = np.array(data_list)
    return neg_data

def scaler(data: np.array, maxvalue, minvalue, drop: bool):
    data_list = []
    for i in range(len(data)):
        if maxvalue >= data[i] >= minvalue:
            data_list.append(data[i])
        elif data[i] > maxvalue:
            if not drop:
                data_list.append(maxvalue)
            else:
                data_list.append(0)
        elif data[i] < minvalue:
            if not drop:
                data_list.append(minvalue)
            else:
                data_list.append(0)
    return np.array(data_list)

def medcouple(data: np.array):
    max_value = data.max()
    min_value = data.min()
    median = np.median(data)
    MC = ((max_value - median) - (median - min_value))/(max_value - min_value)
    return MC

stock_data_file = 'C:/Project/LAB/Normalization/USDRUB++.csv'
stock_data = pd.read_csv(stock_data_file)

volume_KF_DF = np.array(stock_data['VolumeKF_DF'], dtype=float)

pos_data = positive_data(volume_KF_DF)
neg_data = negative_data(volume_KF_DF)

print(f'MAX value: {volume_KF_DF.max()}')
print(f'MIN value: {volume_KF_DF.min()}')
print(f'MEDIAN positive value: {np.median(pos_data)}')
#print(f'MEDIAN negative value: {np.median(neg_data)}')
print(f'25% percentille (pos value): {np.percentile(volume_KF_DF, 25)}')
print(f'75% percentille (pos value): {np.percentile(volume_KF_DF, 75)}')
pos_IQR = np.percentile(volume_KF_DF, 75) - np.percentile(volume_KF_DF, 25)

print(f'IQR: {pos_IQR}')
min_value = np.percentile(volume_KF_DF, 25) - 3 * pos_IQR
max_value = np.percentile(volume_KF_DF, 75) + 3 * pos_IQR
print(f'MAX Value: {max_value}; MIN Value: {min_value}')

""""
mc = medcouple(volume_KF_DF)
print(f'MC: {mc}')
if mc >= 0:
    min_value = np.percentile(volume_KF_DF, 25) - 1.5 * np.exp((-4) * mc) * pos_IQR
    max_value = np.percentile(volume_KF_DF, 75) + 1.5 * np.exp(3 * mc) * pos_IQR
else:
    min_value = np.percentile(volume_KF_DF, 25) - 1.5 * np.exp((-3) * mc) * pos_IQR
    max_value = np.percentile(volume_KF_DF, 75) + 1.5 * np.exp(4 * mc) * pos_IQR
print(f'MAX2 Value: {max_value}; MIN2 Value: {min_value}')
"""

volume_KF_DF_mean = volume_KF_DF.mean()
volume_KF_DF_std = volume_KF_DF.std()

z_data_scale = (volume_KF_DF - volume_KF_DF_mean) / volume_KF_DF_std 

data_scale = scaler(volume_KF_DF, max_value, min_value, drop=False)

#data_scale = -1 + ((data_scale - data_scale.min())/(data_scale.max() - data_scale.min())) * (1 - (-1))
data_scale = (data_scale - data_scale.min())/(data_scale.max() - data_scale.min())


#plt.hist(volume_KF_DF, bins=50)
#plt.hist(data_scale, bins=50)
#plt.show()


lastprice_KF = np.array(stock_data['LastPriceKF'], dtype=float)

lastprice_IQR = np.percentile(lastprice_KF, 75) - np.percentile(lastprice_KF, 25)

print(f'lastprice IQR: {lastprice_IQR}')
min_value = np.percentile(lastprice_KF, 25) - 3 * lastprice_IQR
max_value = np.percentile(lastprice_KF, 75) + 3 * lastprice_IQR
print(f'MAX Value: {max_value}; MIN Value: {min_value}')

data_scale = scaler(lastprice_KF, max_value, min_value, drop=False)
data_scale = (data_scale - data_scale.min())/(data_scale.max() - data_scale.min())
#z_lastprice_KF = (lastprice_KF - lastprice_KF.mean())/lastprice_KF.std()
plt.hist(data_scale, bins=50)
plt.show()