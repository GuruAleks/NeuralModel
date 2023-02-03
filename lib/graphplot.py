import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


STEP = 1


def create_time_steps(length):
    return list(range(-length, 0))


# Отрисовка графика на одну точку вперед
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


# Отрисовка графика несколько точек вперед
def multi_step_plot(history, true_future, prediction):
    #plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 0]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
            label='True Future')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'blue',
            label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                label='Predicted Future')
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'red',
                label='Predicted Future')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()