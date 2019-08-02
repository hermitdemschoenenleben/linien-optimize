import time
import pickle
import random
import numpy as np
from time import sleep
from scipy.signal import savgol_filter, resample

from linien.communication.client import BaseClient
from linien.client.connection import MHz, Vpp
from matplotlib import pyplot as plt

#Test scenarios
from automatix.dev.test_problems.rastrigin import Rastrigin_Scenario

#Optimizer
from automatix.optimizer.optimizer import Optimizer
from automatix.optimizer.engines.cma_es import CMAES
from automatix.optimizer.engines.nelder_mead import NelderMead
from automatix.optimizer.engines.pso import PSO


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_max_diff(array, interesting_size=200):
    points_mean = interesting_size / 10
    resample_factor = int(len(array) / points_mean)
    filtered = resample(array, resample_factor)
    #filtered = savgol_filter(e, 51, 3)
    #filtered = running_mean(e, 51)
    #plt.plot(filtered)
    #plt.plot(np.diff(filtered))
    #plt.plot(np.gradient(filtered))
    #plt.show()
    return np.max(np.abs(filtered))


#main test function
def optimize(connection):
    iteration_count = 10

    s = Rastrigin_Scenario()
    #optimizers = [
    #    ["Nelder-Mead", NelderMead],
    #    ["CMA-ES", CMAES],
    #    ["PSO", PSO],
    #]

    opt = CMAES()

    opt.x0 = [5 * MHz, 1 * Vpp, 0]
    opt._upper_limits = [10 * MHz, 2 * Vpp, 360]
    opt._lower_limits = [0.1 * MHz, 0.01 * Vpp, 0]

    fitness_arr = []

    for iteration in range(iteration_count):
        params = opt.request_parameter_set()
        frequency, amplitude, phase = params

        connection.parameters.modulation_frequency.value = frequency
        connection.parameters.modulation_amplitude.value = amplitude
        connection.parameters.demodulation_phase_a.value = phase

        connection.connection.root.write_data()
        sleep(.5)
        fitness = 1 / (get_max_diff(pickle.loads(connection.parameters.to_plot.value)['error_signal_1']))
        fitness_arr[iteration] = fitness
        opt.insert_fitness_value(fitness, params)

    xFinal = opt.request_results()[0] #gather best result

    plt.plot(fitness_arr)
    plt.show()


c = BaseClient('rp-f0685a.local', 18862, False)
#v = pickle.loads(c.parameters.to_plot.value)
#e = v['error_signal_1']
optimize(c)
#
# plt.plot(e)
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