import numpy as np
from tqdm import tqdm


def univariate_data_old(dataset, start_index, end_index, history_size, target_size):
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


# Нормализация данных
def dataset_normalization(dataset):
    print('Нормализация данных:')
    
    # Выделяем масштабируемые данные
    value = np.array(dataset[1:,0])
    
    # Выполняем нормализацию
    #sklearn_scaler = StandardScaler()
    #value_scalable = sklearn_scaler.fit_transform(value)
    value_scalable = (value-value.mean())/(value.std())
    #print(f'value_scalable: {value_scalable}')
    """
    value_scalable = -1 + ((value - value.min())/(value.max() - value.min())) * (1 - (-1))
    """

    #pos_IQR = np.percentile(value, 75) - np.percentile(value, 25)
    #min_value = np.percentile(value, 25) - 3 * pos_IQR
    #max_value = np.percentile(value, 75) + 3 * pos_IQR
    
    # Приводим данные в диапазон -1...+1
    #value_scale = scaler(value, max_value, min_value, drop=False)
    #value_scalable = -1 + ((value_scale - value_scale.min())/(value_scale.max() - value_scale.min())) * (1 - (-1))

    # Формируем выходные данные
    value_scalable = value_scalable.reshape((value_scalable.shape[0],1))
    time = np.array(dataset[1:,-1])
    time = time.reshape(time.shape[0],1)
    data = np.append(value_scalable, time, axis=1)
    return data


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
            #print(simple_vector[0:history_size])
            out_data.append(simple_vector[0:history_size])
            out_labels.append(simple_vector[history_size:history_size+target_size][0][0])
            tprogress.update()
    return np.array(out_data), np.array(out_labels)

