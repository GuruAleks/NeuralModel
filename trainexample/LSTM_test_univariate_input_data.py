
# Программа тестирования LSTM-нейросети для прогнозирования на основе для одиночных данных
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # Отключаем CUDA-библиотеку

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMNotebookCallback

from tqdm.keras import TqdmCallback
progress = TqdmCallback()

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from keras import initializers

import random

np.set_printoptions(edgeitems=4)

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# Подготовка общих данных
# PARAM:
#   dataset(pd.DataFrame)   - массив данных(в формате DataFrame);
#   history_size(int)       - размер исторических данных;
#   target_size(int)        - размер прогнозируемых данных;
#   shuffle(boolean)        - перемешивание данных (в случайном порядке)  
def reform_data(dataset, history_size: int, target_size: int, shuffle: bool=False) -> np.array:
    # Поготовка к обработке. Определение размеров массивов
    start_index = history_size
    end_index = len(dataset) - (target_size - 1)
    
    # массив обрабатываемых данных
    data = []
    for i in tqdm(range(start_index, end_index), ncols=100, desc='Подготовка данных'):
        work_data = dataset[(i-history_size):(i+target_size),:]
        # Проверяем, если в выборке есть дельта времени более 30 сек, то игнорируем данные

        if (np.all(work_data[:,-1] < 30.0)) and (np.all(work_data[:,0] != 0.0)): 
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(work_data[:, 0], ((history_size+target_size), 1)))
        
    # Формируем одномерный массив-вектор
    dataset_vector = np.array(data)
    # Перемешиваем массив в случайном порядке (если выбрана опция 'shuffle')
    if shuffle:
        print('Перемешиваем массив...')
        np.random.seed(120)
        np.random.shuffle(dataset_vector)
    return dataset_vector


# Подготовка данных для нейросети
# PARAM:
#   dataset(np.Array)   - массив данных(в формате DataFrame);
#   history_size(int)       - размер исторических данных;
#   target_size(int)        - размер прогнозируемых данных;
def preparate_data(dataset, history_size: int, target_size: int):
    # Готовим выходные данные
    out_data = []
    out_labels = []
    with tqdm(total=dataset.shape[0], ncols=100, desc='Подготовка выходных данных') as tprogress:
        for simple_vector in dataset:
            out_data.append(simple_vector[0:history_size])
            out_labels.append(simple_vector[history_size:history_size+target_size][0][0])
            tprogress.update()
    return np.array(out_data), np.array(out_labels)


# Подготовка массива данных в формате прогнозирования
def univariate_data(dataset, start_index, end_index, history_size, target_size, shuffle=False):

    # Поготовка к обработке. Определение размеров массивов
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - (target_size - 1)

    data = []
    for i in tqdm.tqdm(range(start_index, end_index), desc='Подготовка данных'):
        work_data = dataset[(i-history_size):(i+target_size),:]
        # Проверяем, если в выборке есть дельта времени более 30 сек, то игнорируем данные
        # if np.any(work_data[:,-1] < 30.0) and np.all(work_data[:,0] != 0):
        if np.any(work_data[:,-1] < 30.0) and np.all(work_data[:,0] != 0): 
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(work_data[:, 0], ((history_size+target_size), 1)))

    # Формируем одномерный массив-вектор
    dataset_vector = np.array(data)
    # Перемешиваем массив в случайном порядке (если выбрана опция 'shuffle')
    if shuffle:
        print('Перемешиваем массив...')
        np.random.seed(100)
        np.random.shuffle(dataset_vector)

    # Готовим выходные данные
    out_data = []
    out_labels = []
    with tqdm.tqdm(total=dataset_vector.shape[0], desc='Подготовка выходных данных') as tprogress:
        for simple_vector in dataset_vector:
            out_data.append(simple_vector[0:history_size])
            out_labels.append(simple_vector[history_size:history_size+target_size][0][0])
            tprogress.update()
    return np.array(out_data), np.array(out_labels), dataset_vector


"""
#Образ-резерв
def univariate_data(dataset, start_index, end_index, history_size, target_size, shuffle=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        temp_dataset = dataset[range(i-history_size, i+target_size+1)]
        # Проверяем, если в выборке есть дельта времени более 30 сек, то игнорируем данные
        if np.any(temp_dataset[:,-1] < 30.0): 
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices, 0], (history_size, 1)))
            labels.append(dataset[i+target_size, 0])
            #print(f'Data: {data}')
            #print(f'Label: {labels}')
    print(f'npdata: {np.array(data)}')
    print(f'nplabels: {np.array(labels)}')
    return np.array(data), np.array(labels)
"""

def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    # print(f'TestPoint /n TimeSteps:{time_steps}; Apriory_data: {plot_data[0].shape[0]}')
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    plt.grid()
    return plt  


# Нормализация данных
def dataset_normalization(dataset):
    print('Нормализация данных:')
    
    # Выделяем масштабируемые данные
    value = np.array(dataset[1:,0])
    
    # Выполняем нормализацию
    #sklearn_scaler = StandardScaler()
    #value_scalable = sklearn_scaler.fit_transform(value)
    #value_scalable = (value-value.mean())/(value.std())
    #print(f'value_scalable: {value_scalable}')
    """
    value_scalable = -1 + ((value - value.min())/(value.max() - value.min())) * (1 - (-1))
    """

    pos_IQR = np.percentile(value, 75) - np.percentile(value, 25)
    min_value = np.percentile(value, 25) - 3 * pos_IQR
    max_value = np.percentile(value, 75) + 3 * pos_IQR
    
    # Приводим данные в диапазон -1...+1
    value_scale = scaler(value, max_value, min_value, drop=False)
    value_scalable = -1 + ((value_scale - value_scale.min())/(value_scale.max() - value_scale.min())) * (1 - (-1))

    # Формируем выходные данные
    value_scalable = value_scalable.reshape((value_scalable.shape[0],1))
    time = np.array(dataset[1:,-1])
    time = time.reshape(time.shape[0],1)
    data = np.append(value_scalable, time, axis=1)
    return data


# Масштабирование
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


def main(df):
    # Подготовливаем данные
    features_considered = ['LastPriceKF', 'dTime']
    uni_data = df[features_considered]
    uni_data.index = df['DateTime']
    uni_data.head()

    uni_data = uni_data.values

    # Формируем выборки для предиктивного анализа
    univariate_past_history = 20    # Берем 25 выборок для предиктивного анализа 0..29
    univariate_future_target = 1    # Количество выборок предиктивного анализа

    STEP = 1

    dataset_norm = dataset_normalization(uni_data)
    plt.hist(dataset_norm[:,0])
    plt.show() 

    uni_data_all = reform_data(dataset=dataset_norm, 
                                history_size=univariate_past_history,
                                target_size=univariate_future_target,
                                shuffle=False)

    print('Подготовка параметров нейросети')
    # Настраиваем параметры компиляции нейросети
    TRAIN_SPLIT = 0.80 # Значение тренировочной выборки (в долях от целого значения = 1)
    TRAIN_SPLIT_ABS = math.floor(len(uni_data_all) * TRAIN_SPLIT)
    
    print(f'dataset len: {len(uni_data_all)}')
    print(f'dataset train shape: {TRAIN_SPLIT_ABS}')
    
    BUFFER_SIZE = 1000000
    EVALUATION_INTERVAL = 1000
    EPOCHS = 50
    BATCH_SIZE = 128

    """
    # Нормализация данных (на тренировочных данных)
    print('Нормализация данных')
    uni_train_max = uni_data_all.max()
    #uni_train_min = uni_data_all[0:TRAIN_SPLIT_ABS].min()
    uni_train_min = 0
    uni_data_norm = (1 - (-1))*((uni_data_all-uni_train_min)/(uni_train_max-uni_train_min)) + (-1)
    # uni_data_norm = (uni_data_all-uni_data_all[:TRAIN_SPLIT_ABS].mean())/(uni_data_all[:TRAIN_SPLIT_ABS].std())
    plt.hist(uni_data_norm[:,:,-1])
    plt.show() 
    """

    # Формируем общий массив данных (для оценки объемов данных)
    x_uni, y_uni = preparate_data(dataset=uni_data_all, 
                                    history_size=univariate_past_history,
                                    target_size=univariate_future_target)

    x_train_uni, x_val_uni, y_train_uni, y_val_uni = train_test_split(x_uni, y_uni, test_size=0.2)
    #x_train_uni, y_train_uni = x_uni[:TRAIN_SPLIT_ABS], y_uni[:TRAIN_SPLIT_ABS]
    #x_val_uni, y_val_uni = x_uni[TRAIN_SPLIT_ABS:], y_uni[TRAIN_SPLIT_ABS:]

    print(f'x_train_uni shape: {x_train_uni.shape}')
    print(f'y_train_uni shape: {y_train_uni.shape}')
    print(f'x_val_uni shape: {x_val_uni.shape}')
    print(f'y_val_uni shape: {y_val_uni.shape}')

    print ('Single window of stockdata history')
    print (x_train_uni[125])
    print ('\n Target stockdata to predict')
    print (y_train_uni[125])

    plot = show_plot([x_train_uni[125], y_train_uni[125]], 0, 'Sample Example')
    plot.show()

    # Устанавливаем "зерно" случайных чисел перемешивания
    tf.random.set_seed(99)

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    # Определяем количество шагов проверки (кратно BATCH_SIZE)
    val_steps = int(x_val_uni.shape[0] // (EPOCHS))

    #print(f'Val univariate shape: {x_val_uni.shape}')
    #print(f'Validation steps: {val_steps}')

    # размер первого слоя нейросети (по количеству входных и выходных элементов)
    lstm_shape = univariate_past_history

    # Создаем структуру нейросети (один внутренний LSTM слой, один выходной слой)
    simple_stock_predict_model = Sequential()
    simple_stock_predict_model.add(LSTM(units=lstm_shape, 
                                        input_shape=x_train_uni.shape[-2:],
                                        activation='tanh', 
                                        return_sequences=True))
    simple_stock_predict_model.add(LSTM(units=int(lstm_shape//2),
                                        activation='tanh'))
    simple_stock_predict_model.add(Dense(1, activation='tanh'))
    
    print(simple_stock_predict_model.summary())

    simple_stock_predict_model.compile(optimizer='adam', loss='mse')

    simple_history = simple_stock_predict_model.fit(
                                x_train_uni, y_train_uni,
                                batch_size=BATCH_SIZE, epochs=EPOCHS,
                                validation_data=(x_val_uni, y_val_uni), 
                                use_multiprocessing=True,
                                workers=4)

    plt.plot(simple_history.history['loss'])
    plt.plot(simple_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
  

    """
    simple_history = simple_stock_predict_model.fit(
                                train_univariate, epochs=EPOCHS,
                                steps_per_epoch=EVALUATION_INTERVAL,
                                validation_data=val_univariate, 
                                validation_steps=val_steps,
                                max_queue_size=1,
                                workers=4, 
                                use_multiprocessing=True
                                )
    """
    """
    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM((univariate_past_history+univariate_future_target), input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1, activation='relu')
        ])
    """

    #tf.keras.utils.plot_model(simple_lstm_model)

    # simple_lstm_model.compile(optimizer='adagrad', loss='mlc', metrics = 'Accuracy')
    # simple_lstm_model.compile(optimizer='adam', loss='mae')

    #for x, y in val_univariate.take(1):
    #    print(simple_lstm_model.predict(x).shape)

    #print(x_train_uni.shape)

    """
    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data=val_univariate, validation_steps=val_steps,
                        max_queue_size=1,
                        workers=4, 
                        use_multiprocessing=True)
    """
    """
    simple_lstm_model.fit(x=x_uni, y=y_uni, epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_split=0.30,
                        workers=4, 
                        use_multiprocessing=True)
    """

    # Рисуем график
    samples = 20
    
    for i in range(samples):
        n = random.randint(0, x_val_uni.shape[0])
        x = x_val_uni[n]
        y = y_val_uni[n]
        print(f'simple_stock_predict_model.predict(x): {simple_stock_predict_model.predict(x)}')
        print(f'simple_stock_predict_model.predict(x)[0]: {simple_stock_predict_model.predict(x)[0]}')
        plot = show_plot([x, y,
                            simple_stock_predict_model.predict(x)[0]], 0, 'Simple LSTM model')
        plot.show()

    """
    for x, y in val_univariate.take(30):
        # print(f'X: {x[0].numpy()}; Y: {y[0].numpy()}; DATA: {simple_lstm_model.predict(x)[0]}')
        plot = show_plot([x[0].numpy(), y[0].numpy(),
                            simple_stock_predict_model.predict(x)[0]], 0, 'Simple LSTM model')
        plot.show()
    """

if __name__ == "__main__":
    
    # Считываем данные:
    usd_stock_data = 'C:/Project/LAB/Normalization/USDRUB++.csv'
    stock_data = pd.read_csv(usd_stock_data)
    main(stock_data)