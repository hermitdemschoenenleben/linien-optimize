import time
import math
import pickle
import random
import numpy as np
from time import sleep, time
from scipy.signal import savgol_filter, resample

from linien.communication.client import BaseClient
from linien.client.connection import MHz, Vpp
from matplotlib import pyplot as plt

from cma_es import CMAES


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_max_diff(array, interesting_size=200):
    points_mean = interesting_size / 10
    resample_factor = int(len(array) / points_mean)
    filtered = resample(array, resample_factor)

    # resampling assumes periodicity
    # --> if the ends don't match (e.g. doppler background) we have very steep
    # edges. Therefore, we crop them.
    N = int(len(filtered) / 8)
    filtered = filtered[N:-N]
    #filtered = savgol_filter(e, 51, 3)
    #filtered = running_mean(e, 51)
    """plt.plot(filtered)
    plt.plot(np.diff(filtered))
    plt.plot(np.gradient(filtered))
    plt.show()"""
    return np.max(np.abs(np.gradient(filtered)))


def optimize(connection):
    iteration_count = 300

    opt = CMAES()
    opt.sigma = .5

    max_freq = 100 * MHz
    max_ampl = 2 * Vpp
    max_phase = 360

    opt.x0 = [0.5, 0.5, 0.5]
    opt._upper_limits = [1, 1, 1]
    opt._lower_limits = [0.01, 0.01, 0]

    fitness_arr = []

    def set_parameters(params):
        frequency, amplitude, phase = params
        print('%.2f MHz, %.2f Vpp, %d deg' % (frequency * max_freq / MHz, amplitude * max_ampl / Vpp, phase * max_phase))
        connection.connection.root.exposed_pause_acquisition()
        connection.parameters.modulation_frequency.value = frequency * max_freq
        connection.parameters.modulation_amplitude.value = amplitude * max_ampl
        connection.parameters.demodulation_phase_a.value = phase * max_phase
        connection.connection.root.write_data()
        connection.connection.root.exposed_continue_acquisition()

    storage = {'data': None}
    def data_received(value):
        storage['data'] = value
    connection.parameters.to_plot.change(data_received)

    for iteration in range(iteration_count):
        print('iteration', iteration)
        params = opt.request_parameter_set()
        set_parameters(params)
        connection.parameters.call_listeners()
        storage['data'] = None
        while not storage['data']:
            connection.parameters.call_listeners()
            sleep(.01)

        #spectrum = pickle.loads(connection.parameters.to_plot.value)['error_signal_1']
        spectrum = pickle.loads(storage['data'])['error_signal_1']

        fitness = math.log(1 / (get_max_diff(spectrum)))
        print('fitness', fitness)
        fitness_arr.append(fitness)
        opt.insert_fitness_value(fitness, params)

    xFinal = opt.request_results()[0] #gather best result
    print('FINAL!', xFinal)
    set_parameters(xFinal)
    sleep(3)
    spectrum = pickle.loads(connection.parameters.to_plot.value)['error_signal_1']
    plt.plot(spectrum)
    plt.show()

    plt.plot(fitness_arr)
    plt.show()


c = BaseClient('rp-f0685a.local', 18862, False)
optimize(c)
asd
v = pickle.loads(c.parameters.to_plot.value)
e = v['error_signal_1']
plt.plot(e)
plt.show()
"""phases = np.linspace(0, 360, 36)
frequencies = np.linspace(1, 15, 15)
data = np.zeros((len(frequencies), len(phases)))

for f_idx, frequency in enumerate(frequencies):
    c.parameters.modulation_frequency.value = frequency * MHz

    for p_idx, phase in enumerate(phases):
        c.parameters.demodulation_phase_a.value = phase
        c.connection.root.write_data()
        sleep(.5)
        data[f_idx, p_idx] = (get_max_diff(pickle.loads(c.parameters.to_plot.value)['error_signal_1']))

from matplotlib import pyplot as plt
ff, pp = np.meshgrid(phases, frequencies)
plt.pcolormesh(ff, pp, data)
#plt.pcolormesh(data)
#plt.plot(phases, result)
plt.show()"""