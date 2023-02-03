from unittest import result
import pandas as pd
import numpy as np
import datetime
#from sklearn.preprocessing import StandardScaler

# Класс для вычисления калмановской фильтрации данных
# ВХОДНЫЕ ПАРАМЕТРЫ:
#   zero_data - нулевые данные для инициализации
class KalmanFilter:
    def __init__(self, zero_data, variance=0.005) -> None:
        self.covariance = 1.00              # коэффициент корреляции в модели движения
        self.dNoise = 1.5                   # дисперсия шума (в будущем сделать вычисляемым значением)
        self.__variance_move = variance     # дисперсия(ошибка) СВ в модели движения (лучше при 0.05)
        self.prev_data = zero_data          # формируем предварительные данные
        self.p_variance = self.dNoise       # инициализируем дисперсию первой оценки

    # вычисление фильтрованного значения
    # ВХОДНЫЕ ПАРАМЕТРЫ:
    #   data - входные данные
    
    def KF_processing(self, data):
        # Вычисляем ошибку предыдущего шага
        Pe = self.p_variance * (self.covariance**2) + (self.__variance_move**2)
        # Вычисляем оценку ошибки текущего шага
        self.p_variance = (Pe * self.dNoise) / (Pe + self.dNoise)
        # Вычисляем значение фильтрованного сигнала с учетом ошибки
        result = self.covariance*self.prev_data + self.p_variance/self.dNoise*(data - self.covariance*self.prev_data)
        self.prev_data = result
        return result

    @property
    def variance_move(self):
        return self.__variance_move

    @variance_move.setter
    def variance_move(self, value):
        self.__variance_move = value


# Расчет производной
# ВХОДНЫЕ ПАРАМЕТРЫ:
#   diff_data   - дифференцируемые данные, массив данных от 2 до 5 элементов
#   dt          - интервал времени, в секундах временное значение
# ВЫХОДНЫЕ ДАННЫЕ:
#   result      - результат расчета производной
def derivative_calc(diff_data: np.array, dt):
    #print('derivative:', len(diff_data), ' diff_data:', diff_data, ' dt:', dt)
    # Вычисление производной по пятиточечному шаблону (в крайней правой точке)
    if len(diff_data) == 5:
        result = (3*diff_data[0] - 16*diff_data[1] + 36*diff_data[2] - 48*diff_data[3] + 25*diff_data[4])/(12*dt)
    # Вычисление производной по четырехточечному шаблону (в крайней правой точке)
    elif len(diff_data) == 4:
        result = (0 - 2*diff_data[0] + 9*diff_data[1] - 18*diff_data[2] + 11*diff_data[3])/(6*dt)
    # Вычисление производной по трехточечному шаблону (в крайней правой точке)
    elif len(diff_data) == 3:
        result = (1*diff_data[0] - 4*diff_data[1] + 3*diff_data[2])/(2*dt)
    # Вычисление производной по двум точкам
    elif len(diff_data) == 2:
        result = (diff_data[1] - diff_data[0])/dt
    else:
        raise ValueError('Неправильно выбраны данные для диффренцирования')
    return result


def derivative(func: list, x: list=0, windowsize: int=2):
   
    # Результат (возвращается список)
    result = []
    
    # Определяем размер окна наблюдения 
    # значения окна - целое число, количество выборок (точек наблюдения)
    window_size = windowsize

    # Проверяем размер окна наблюдение на соответствие размеру наблюдаемой функции
    if window_size > len(func):
        window_size = len(func)
        result = func
        return result

    # Флаг контроля инициализации запуска расчета производной
    init_status = True

    # Перебираем элементы массива
    for cur_pos in range(window_size, len(func)+1, 1):
        # cur_pos - указатель текущего положения
        # Заполняем начальные позиции массива (нулями, поскольку отсутствуют данные для расчета производной)
        if init_status:
            result = [0 for i in range(0, window_size-1)]
            init_status = False
   
        # Промежуточный массив данных (срез выборок)
        stockdata_slice = func[cur_pos-window_size:cur_pos:1].tolist()
        # dt1 = x[cur_pos-2] - x[cur_pos-3]
        
        t2 = datetime.datetime.strptime(x[cur_pos-1], '%Y-%m-%d %H:%M:%S').timestamp()
        t1 = datetime.datetime.strptime(x[cur_pos-2], '%Y-%m-%d %H:%M:%S').timestamp()
        dt = t2 - t1
        if dt == 0:
            print('t2=', x[cur_pos-1], 't1=', x[cur_pos-2])
            dt = 1

        # считываем данные для расчета производной (в размере окна дифференциорования)
        # result.append(df_dt_3p(stockdata_slice, dt1, dt2))
        result.append(derivative_calc(stockdata_slice, dt))
    return result




# Расчитываем разницу времени
def dTime(time_prev: np.datetime64, time_cur: np.datetime64) -> float:
    #t2 = datetime.datetime.strptime(time_cur, '%Y-%m-%d %H:%M:%S').timestamp()
    #t1 = datetime.datetime.strptime(time_prev, '%Y-%m-%d %H:%M:%S').timestamp()
    #result = t2 - t1
    return (time_cur - time_prev)/np.timedelta64(1, "s")