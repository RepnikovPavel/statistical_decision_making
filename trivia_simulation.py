import numpy as np
import matplotlib.pyplot as plt
from support_functions import eval_list_of_integrals
import time

def simulate_one_car(old_z, old_omega, list_for_integrate, new_omega_list, list_of_integrals_z_di,plot_simul,block_canvas):
    start_time = time.time()
    dist_to_line = 360
    T = 50
    step = 0.01
    N = int(T / step)
    t_vec = np.linspace(0, T, N)
    x_vec = np.zeros(N, )  # m
    v_vec = np.zeros(N, )  # m/s
    a_vec = np.zeros(N, )  # m/s^2
    obs = np.zeros(2, )
    last_index_for_plot = 0
    distanses_to_line = np.zeros(N, )
    for i in range(N - 1):
        print("\rtime {:.2f} of {:.2f}".format(step*i, T), end='')
        # взяли наблюдения
        obs[0] = dist_to_line - x_vec[i]
        obs[1] = v_vec[i]
        distanses_to_line[i] = dist_to_line - x_vec[i]
        if obs[1] > 22.2:
            obs[1] = 22.2

        if obs[1] < 0:
            last_index_for_plot = i
            break

        if obs[0] <= 0:
            last_index_for_plot = i
            break

        if obs[1] > 22.2:
            obs[1] = 22.2
        # получили ответ
        a_i = eval_list_of_integrals(obs, old_z, old_omega, list_for_integrate, new_omega_list, list_of_integrals_z_di)[
                  0]
        a_vec[i + 1] = a_i
        # промоделировали
        v_vec[i + 1] = v_vec[i] + (step * (a_i + a_vec[i]) / 2)
        x_vec[i + 1] = x_vec[i] + step * ((v_vec[i + 1] + v_vec[i]) / 2)
    stop_time = time.time()
    print("\ntime of simul {:.2f}".format(stop_time - start_time))
    if plot_simul == True:
        fig, ax = plt.subplots(1, 1)
        if last_index_for_plot == 0:
            last_index_for_plot = N
        # ax[0].plot(x_vec[:last_index_for_plot], v_vec[:last_index_for_plot])
        # ax[0].set_xlabel("пройденное расстояние, м")
        # ax[0].set_ylabel("v, м/c")
        # ax[0].set_title("растояние до стоп линии {} м".format(dist_to_line))

        # ax[2].plot(t_vec[:last_index_for_plot], distanses_to_line[:last_index_for_plot])
        # ax[2].set_xlabel("t, c")
        # ax[2].set_ylabel("x, м")

        ax.plot(t_vec[:last_index_for_plot], a_vec[:last_index_for_plot])
        ax.set_xlabel(r'$время \:симуляции \:t, \:c$')
        ax.set_ylabel(r'$ускорение \: y_{1}, \: м\cdot c^{-2} $')
        ax.grid(True, 'both', 'both')
        plt.show(block=block_canvas)

    return distanses_to_line[:last_index_for_plot], v_vec[:last_index_for_plot]
