
# Программа тестирования LSTM-нейросети для прогнозирования на основе множества данных

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # Отключаем CUDA-библиотеку

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import preprocessing


TRAIN_SPLIT = 1711000
tf.random.set_seed(13)

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

BATCH_SIZE = 17110
BUFFER_SIZE = 1000000

EVALUATION_INTERVAL = 1000
EPOCHS = 100

STEP = 1

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        
        temp_dataset = dataset[range(i-history_size, i+target_size, step)]
        if np.any(temp_dataset[:,-1] < 30.0): # Проверяем, если в выборке есть дельта времени более 30 сек, то игнорируем данные
            data.append(dataset[indices][:,[0,1,2]])
            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])
    return np.array(data, dtype='float'), np.array(labels, dtype='float')


def create_time_steps(length):
    return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 0]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
            label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                label='Predicted Future')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def dataset_normalization(data):
    DateTime = np.array(data.index, dtype='datetime64')
    dTime = np.array(data['dTime'], dtype='float')
    
    lastprice = np.array(data['LastPriceKF'], dtype='float')
    lastprice_min = lastprice.min()
    lastprice_max = lastprice.max()
    lastprice_norm = (lastprice - lastprice_min)/(lastprice_max-lastprice_min)

    lastpriceKF_DF = np.array(data['LastPriceKF_DF'], dtype='float')
    lastpriceKF_DF_mean = lastpriceKF_DF.mean()
    lastpriceKF_DF_std = lastpriceKF_DF.std()
    lastpriceKF_DF_norm = (lastpriceKF_DF - lastpriceKF_DF_mean)/lastpriceKF_DF_std

    volumeKF_DF = np.array(data['VolumeKF_DF'], dtype='float')
    volumeKF_DF_mean = volumeKF_DF.mean()
    volumeKF_DF_std = volumeKF_DF.std()
    volumeKF_DF_norm = (volumeKF_DF - volumeKF_DF_mean)/volumeKF_DF_std
    
    result = pd.DataFrame({'DateTime': DateTime.tolist(),
                        'LastPriceKF': lastprice_norm.tolist(),
                        'LastPriceKF_DF': lastpriceKF_DF_norm.tolist(),
                        'VolumeKF_DF': volumeKF_DF_norm.tolist(),
                        'dTime': dTime.tolist()})
    
    result.set_index('DateTime', inplace=True)
    return result


def main(df):

    # Подготавливаем данные
    features_considered = ['LastPriceKF', 'LastPriceKF_DF', 'VolumeKF_DF', 'dTime']
    features = df[features_considered]
    features.index = df['DateTime']
    features.head()

    # Выполняем нормализацию
    features_norm = dataset_normalization(features)
    features_norm.head()
    #dataset = features.values
    dataset = features_norm.values
    #print(f'DataSet: {dataset}')
    #print(f'DataSet_mem: {dataset[:, -1]}')
    #data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    #data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    #dataset = (dataset-data_mean)/data_std

    past_history = 96
    future_target = 24

    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                                TRAIN_SPLIT, None, past_history,
                                                future_target, STEP)

    #print(f'143_x_train_multi_shape:{x_train_multi.shape}')
    #print(f'144_x_val_multi_shape:{x_val_multi.shape}')

    print (f'Multi window of past history : {x_train_multi[0].shape}')
    print (f'Target stockdata to predict : {y_train_multi[0].shape}')

    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    for x, y in train_data_multi.take(1):
        multi_step_plot(x[0], y[0], np.array([0]))


    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(96,
                                            return_sequences=True,
                                            input_shape=x_train_multi.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(24, activation='relu'))
    multi_step_model.add(tf.keras.layers.Dense(24))

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    for x, y in val_data_multi.take(1):
        print (f'multi_step_model.predict(x).shape:{multi_step_model.predict(x).shape}')

    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=25)
    

    plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

    for x, y in val_data_multi.take(10):
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])


if __name__ == "__main__":
    #csv_path = 'D:/Work/PROGR/BIG_DATA/TradeModel/TradeModel-1/jena_climate_2009_2016.csv''D:/Work/PROGR/BIG_DATA/DATA/USDRUB+++.csv'
    csv_path = 'D:/Work/PROGR/BIG_DATA/DATA/USDRUB+++.csv'
    df = pd.read_csv(csv_path)
    main(df)