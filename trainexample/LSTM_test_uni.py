
# Программа тестирования LSTM-нейросети для прогнозирования на основе для одиночных данных
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # Отключаем CUDA-библиотеку

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

TRAIN_SPLIT = 300000
tf.random.set_seed(13)

# Настраиваем параметры компиляции нейросети
BATCH_SIZE = 256
BUFFER_SIZE = 10000

EVALUATION_INTERVAL = 2000
EPOCHS = 100

STEP = 1

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# Подготовка массива данных в формате прогнозирования
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
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
    #plt.ylim([-2.11, -1.94])
    plt.grid()
    # plt.show()
    return plt  


def main(df):
    
    # Подготовливаем данные
    uni_data = df['T (degC)']
    uni_data.index = df['Date Time']
    uni_data.head()
    # uni_data.plot(subplots=True)
    uni_data = uni_data.values

    # Проводим нормализацию
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()
    uni_data = (uni_data-uni_train_mean)/uni_train_std
    
    # Формируем выборки для предиктивного анализа
    univariate_past_history = 50    # Берем 20 выборок для предиктивного анализа 0..19
    univariate_future_target = 0    # Одна выборка предиктивного анализа

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                            univariate_past_history,
                                            univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                        univariate_past_history,
                                        univariate_future_target)
    
    print ('Single window of past history')
    print (x_train_uni[0])
    print ('\n Target temperature to predict')
    print (y_train_uni[0])
    
    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])

    # simple_lstm_model.compile(optimizer='adagrad', loss='mlc', metrics = 'Accuracy')
    simple_lstm_model.compile(optimizer='adam', loss='mse', metrics = 'Hinge')

    for x, y in val_univariate.take(1):
        print(simple_lstm_model.predict(x).shape)

    print(x_train_uni.shape)

    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data=val_univariate, validation_steps=50)
    
    for x, y in val_univariate.take(30):
        plot = show_plot([x[0].numpy(), y[0].numpy(),
                            simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
        plot.show()


if __name__ == "__main__":
    
    # Считываем данные:
    csv_path = 'D:/Work/PROGR/BIG_DATA/TradeModel/TradeModel-1/jena_climate_2009_2016.csv'
    df = pd.read_csv(csv_path)
    main(df)