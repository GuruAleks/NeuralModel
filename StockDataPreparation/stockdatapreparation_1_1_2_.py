# -*- coding: utf-8 -*-
# Формирование данных модели
import asyncio
import numpy as np
import pandas as pd
from tqdm import tqdm

from math import sin, cosh

#from decimal import ROUND_FLOOR, Decimal

import sys

sys.path.append('C:/Project/StockTrain')

#import mathtool
from lib.mathtool import KalmanFilter


__version__ = '1.1.2'
# Убраны мало влияющие на конечный результат значения
# При паузах введено сброс параметров фильтров

# Предельное время после которого обнуляется настройки (в секундах)
SET_CRITICAL_DTIME = 60.0

LASTPRICE_KF_VARIANCE = 0.085

LASTPRICE_DEVIATION_VARIANCE = 0.010

VOLUMESUM_KF_VARIANCE = 0.1

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
    
    # Массив значений приближения к нулевой оси(вычисление дивергенции)
    lastprice_deviation = np.zeros(datetime_arr.shape[0], dtype=float)
    
    lastprice_KF[0] = lastprice_arr[0]  # формируем "ненулевое" значение первой выборки
    lastprice_deviation[0] = lastprice_arr[0]

    # Создаем экзмепляр класса Фильтра Калмана (в качестве инициализации начальных значений применяется первое значение изменения цены)
    lastprice_class_KF = KalmanFilter(lastprice_arr[0], variance=LASTPRICE_KF_VARIANCE)
    lastprice_class_deviation = KalmanFilter(lastprice_arr[0], variance=LASTPRICE_DEVIATION_VARIANCE)

    for cnt in tqdm(range(1, datetime_arr.shape[0]), desc='Last Price Calc'):
        # Делаем проверку задержки времени
        if deltatime_arr[cnt] <= SET_CRITICAL_DTIME:
            # если меньше 30 секунд, то расчитываем по фильтру Калмана
            lastprice_KF[cnt] = lastprice_class_KF.KF_processing(lastprice_arr[cnt])
            lastprice_deviation[cnt] = lastprice_class_deviation.KF_processing(lastprice_arr[cnt])
        else:
            # если больше 30 секунд, то обнуляем начальное состояние фильтра Калмана
            lastprice_KF[cnt] = lastprice_arr[cnt]
            lastprice_deviation[cnt] = lastprice_arr[cnt]
            lastprice_class_KF.KF_reset(lastprice_arr[cnt])
            lastprice_class_deviation.KF_reset(lastprice_arr[cnt])
        await asyncio.sleep(0)

    lastprice_div = lastprice_KF - lastprice_deviation
    
    # Создаем массив полученных данных в формате DataFrame
    result = {'LastPriceKF': lastprice_KF,
                'LastPriceDeviation': lastprice_deviation,
                'LastPriceDIV': lastprice_div}

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
   
    # Инициализация начальных значений
    volume_sum[0] = volume_arr[0]
    
    # Создаем экзмепляр класса Фильтра Калмана (в качестве инициализации начальных значений применяется первое значение изменения цены)
    volume_class_KF = KalmanFilter(volume_sum[0], variance=VOLUMESUM_KF_VARIANCE)

    for cnt in tqdm(range(1, datetime_arr.shape[0]), desc='Volume Calc'):
        #volume_sum[cnt] += volume_arr[cnt]
        #volume_KF[cnt] = volume_class_KF.KF_processing(volume_sum[cnt])
        if deltatime_arr[cnt] <= SET_CRITICAL_DTIME:
            volume_sum[cnt] = volume_sum[cnt-1] + volume_arr[cnt]
            volume_KF[cnt] = volume_class_KF.KF_processing(volume_sum[cnt])
        else:
            volume_sum[cnt] = 0
            volume_KF[cnt] = 0
            volume_class_KF.KF_reset(volume_sum[cnt])
        #volume_bar.next()
        await asyncio.sleep(0)

    # Создаем массив полученных данных в формате DataFrame
    result = {'VolumeSUM': volume_sum,
                'VolumeKF': volume_KF}

    return result


async def model_calc(stock_data_file):

    print('-= START CALC =-')

    # Формируем массив записей времени
    stockdata = pd.read_csv(stock_data_file)
    lastpricecalc, volumecalc = await asyncio.gather(last_price_calc(stockdata), volume_calc(stockdata))

    result = pd.DataFrame({'DateTime': stockdata['DateTime'],
                            'LastPrice': stockdata['LastPrice'],
                            'LastPriceKF': lastpricecalc['LastPriceKF'],
                            'LastPriceDEV': lastpricecalc['LastPriceDeviation'],
                            'LastPriceDIV': lastpricecalc['LastPriceDIV'],
                            'Volume': stockdata['Volume'],
                            'VolumeSUM': volumecalc['VolumeSUM'],
                            'VolumeKF': volumecalc['VolumeKF'],
                            'dTime': stockdata['dTime']})
    return result
    
if __name__ == '__main__':
    
    # количество бумаг в одном лоте
    stock_data_file = 'C:/Project/data/usdrub_original.csv'
    result = asyncio.run(model_calc(stock_data_file))
    result.to_csv(f'C:/Project/data/usdrub_{__version__}.csv', index=False)
    
