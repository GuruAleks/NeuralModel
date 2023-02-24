# -*- coding: utf-8 -*-
# Формирование данных модели
import asyncio
import numpy as np
import pandas as pd
from tqdm import tqdm

from math import sin, cosh

#from decimal import ROUND_FLOOR, Decimal

from lib.mathtool import KalmanFilter, dTime, derivative, derivative_calc


__version__ = '1.0.1'


# Вычисляем в асинхронном потоке параметры значений изменений цены
async def last_price_calc(stockdata: pd.DataFrame) -> pd.DataFrame:
    """
    ВХОДНЫЕ ПАРАМЕТРЫ:
    - stockdata         - DataFrame-массив данных
    - datetime_arr     - numpy-массив дата-время;
    - lastprice_arr    - numpy-массив изменения цены;

    ВЫХОДНЫЕ ПАРАМЕТРЫ (в формате pd.dataframe):
    - lastprice_KF     - изменение цены, пропущенное через фильтр Калмана;
    - lastprice_KF_DF  - производная изменения цены, пропущенное через фильтр Калмана;
    """

    datetime_arr = np.array(stockdata['DateTime'], dtype='datetime64')
    lastprice_arr = np.array(stockdata['LastPrice'], dtype=float)
    deltatime_arr = np.array(stockdata['dTime'], dtype=float)


    # создаем болванки(пустые массивы)
    # - массив изменения цены, пропущенного через фильтр Калмана;
    lastprice_KF = np.zeros(datetime_arr.shape[0], dtype=float)
    
    # - массив производной изменения цены, пропущенного через фильтр Калмана;
    lastprice_KF_DF = np.zeros(datetime_arr.shape[0], dtype=float)
    
    # Создаем экзмепляр класса Фильтра Калмана (в качестве инициализации начальных значений применяется первое значение изменения цены)
    lastprice_class_KF = KalmanFilter(lastprice_arr[0], variance=0.075)

    for cnt in tqdm(range(1, datetime_arr.shape[0]), desc='Last Price Calc'):
        lastprice_KF[cnt] = lastprice_class_KF.KF_processing(lastprice_arr[cnt])
        if cnt >= 2:
            # Вычисляем производную от изменения цены - берем производную по трем точкам.
            # время дифференцирования усредняем 
            lastprice_KF_DF[cnt] = derivative_calc(lastprice_KF[(cnt-2):(cnt+1)], np.mean(deltatime_arr[(cnt-2) : (cnt+1)]))
        else:
            # Для первых двух значений заполняем по умолчанию [0]..[1]
            lastprice_KF_DF[cnt] = derivative_calc(lastprice_KF[:cnt+1], deltatime_arr[cnt])    
        await asyncio.sleep(0)

    # Создаем массив полученных данных в формате DataFrame
    result = pd.DataFrame({'DateTime': datetime_arr.tolist(),
                            'LastPriceKF': lastprice_KF.tolist(),
                            'LastPriceKF_DF': lastprice_KF_DF.tolist()})
    
    result = result.set_index(['DateTime'], drop=False)

    return result


# Вычисляем в асинхронном потоке параметры значений изменений объема сделок
async def volume_calc(stockdata: pd.DataFrame) -> pd.DataFrame:
    """
    ВХОДНЫЕ ПАРАМЕТРЫ:
    - stockdata         - DataFrame-массив данных
    - datetime_arr      - numpy-массив дата-время;
    - volume_arr        - numpy-массив изменения цены;

    ВЫХОДНЫЕ ПАРАМЕТРЫ (в формате pd.dataframe):
    - volume            - изменение суммы объема нарастающим итогом
    - volume_KF         - изменение суммы объема нарастающим итогом, пропущенное через фильтр Калмана;
    - volume_KF_DF      - производная от изменения суммы объема с нарастающим итогом, пропущенное через фильтр Калмана;
    """

    datetime_arr = np.array(stockdata['DateTime'], dtype='datetime64')
    volume_arr = np.array(stockdata['Volume'], dtype=float)
    deltatime_arr = np.array(stockdata['dTime'], dtype=float)

    # создаем болванки(пустые массивы)
    # - массив объема суммированного нарастающим итогом;
    volume_sum = np.zeros(datetime_arr.shape[0], dtype=float)
    # - массив изменения объема, пропущенного через фильтр Калмана;
    volume_KF = np.zeros(datetime_arr.shape[0], dtype=float)
    # - массив производной изменения объема, пропущенного через фильтр Калмана;
    volume_KF_DF = np.zeros(datetime_arr.shape[0], dtype=float)
    
    # Инициализация начальных значений
    volume_sum[0] = volume_arr[0]
    
    # Создаем экзмепляр класса Фильтра Калмана (в качестве инициализации начальных значений применяется первое значение изменения цены)
    volume_class_KF = KalmanFilter(volume_sum[0], variance=0.1)

    for cnt in tqdm(range(1, datetime_arr.shape[0]), desc='Volume Calc'):
        volume_sum[cnt] += volume_arr[cnt]
        volume_KF[cnt] = volume_class_KF.KF_processing(volume_sum[cnt])
        if cnt >= 2:
            # Вычисляем производную от изменения цены - берем производную по трем точкам.
            # время дифференцирования усредняем 
            volume_KF_DF[cnt] = derivative_calc(volume_KF[(cnt-2):(cnt+1)], np.mean(deltatime_arr[(cnt-2) : (cnt+1)]))
        else:
            # Для первых двух значений заполняем по умолчанию [0]..[1]
            volume_KF_DF[cnt] = derivative_calc(volume_KF[:cnt+1], deltatime_arr[cnt])    
        #volume_bar.next()
        await asyncio.sleep(0)

    # Создаем массив полученных данных в формате DataFrame
    result = pd.DataFrame({'DateTime': datetime_arr.tolist(),
                            'VolumeSUM': volume_sum.tolist(),
                            'VolumeKF': volume_KF.tolist(),
                            'VolumeKF_DF': volume_KF_DF.tolist()})
    result = result.set_index(['DateTime'], drop=False)

    return result


async def model_calc(stock_data_file):

    print('-= START CALC =-')

    # Формируем массив записей времени
    stockdata = pd.read_csv(stock_data_file)
    lastpricecalc, volumecalc = await asyncio.gather(last_price_calc(stockdata), volume_calc(stockdata))

    #lastpricecalc.to_csv('C:/Project/stock_graph/util/StockDataPreparationUtil/USDRUB_lastprice.csv')
    #volumecalc.to_csv('C:/Project/stock_graph/util/StockDataPreparationUtil/USDRUB_volume.csv')
    datetime_arr = np.array(stockdata['DateTime'], dtype='datetime64')
    lastprice_arr = np.array(stockdata['LastPrice'], dtype=float)
    deltatime_arr = np.array(stockdata['dTime'], dtype=float)
    volume_arr = np.array(stockdata['Volume'], dtype=float)
    volume_sum = np.array(volumecalc['VolumeSUM'], dtype=float)
    volume_KF = np.array(volumecalc['VolumeKF'], dtype=float)
    volume_KF_DF = np.array(volumecalc['VolumeKF_DF'], dtype=float)                        
    lastprice_KF = np.array(lastpricecalc['LastPriceKF'], dtype=float)
    lastprice_KF_DF = np.array(lastpricecalc['LastPriceKF_DF'], dtype=float)

    result = pd.DataFrame({'DateTime': datetime_arr.tolist(),
                            'LastPrice': lastprice_arr.tolist(),
                            'LastPriceKF': lastprice_KF.tolist(),
                            'LastPriceKF_DF': lastprice_KF_DF.tolist(),
                            'Volume': volume_arr.tolist(),
                            'VolumeSUM': volume_sum.tolist(),
                            'VolumeKF': volume_KF.tolist(),
                            'VolumeKF_DF': volume_KF_DF.tolist(),
                            'dTime': deltatime_arr.tolist()})

    #print('result:', result)
    return result
    
if __name__ == '__main__':
    
    # количество бумаг в одном лоте
    stock_data_file = 'C:/Project/data/usdrub_original.csv'

    #stock_data = pd.read_csv(stock_data_file)

    #result = model_calc(stock_data)
    result = asyncio.run(model_calc(stock_data_file))
    result.to_csv('C:/Project/data/usdrub_work.csv', index=False)
    
