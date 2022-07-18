from support_functions import compute_f
from support_functions import my_loss
from support_functions import plot_f
from support_functions import train_and_save_z
from support_functions import plot_consistency
from support_functions import get_indicies
from support_functions import get_output
from support_functions import make_list_of_rect_from_rules
from support_functions import make_tensor_from_list_of_rects
from support_functions import make_new_omega_for_rect
from support_functions import intergate_list_of_rules_on_tmp_new_omega
from support_functions import eval_list_of_integrals
from support_functions import plot_x_2_x_3_use_eval_list
from support_functions import compute_f_x
from support_functions import torch_to_numpy_z

import os
import imageio

from support_functions import clear_dir_for_chat_with_sim

import torch
import numpy as np
import matplotlib.pyplot as plt

import time


def make_omega_and_np_omega(rules):
    # прочитаем разбиение(рисочки) из словаря с правилами и сделаем отрезки из рисочек разбиения
    omega = []
    np_omega = []
    for rule in rules:
        omega.append([])
        np_omega.append([])
        for distribution in rules[rule]:
            vec_of_omega_k = []
            np_vec_of_omega_k = []
            for i in range(rules[rule][distribution]["N"]):
                a = rules[rule][distribution]["razb"][i]
                b = rules[rule][distribution]["razb"][i + 1]
                tmp_segment = torch.zeros(size=(2,), requires_grad=False)
                tmp_segment[0] = a
                tmp_segment[1] = b
                vec_of_omega_k.append(tmp_segment)
                np_tmp_segment = np.zeros(2,)
                np_tmp_segment[0] = a
                np_tmp_segment[1] = b
                np_vec_of_omega_k.append(np_tmp_segment)
            np_omega[rule-1].append(np_vec_of_omega_k)
            omega[rule - 1].append(vec_of_omega_k)
    for i in range(len(np_omega)):
        for j in range(len(np_omega[i])):
            np_omega[i][j] = np.asarray(np_omega[i][j])
    for i in range(len(np_omega)):
        np_omega[i] = np.asarray(np_omega[i], dtype="object")

    return omega, np_omega


def take_gradients(num_of_rules, dimension,rules, omega, train_dict, a, h, f, coeff_list,init_z,distrib_of_rules,
                   check_target_values,
                   min_cons,
                   min_norm,
                   min_distr,
                   previous_z_lists,
                   problems_areas_info,
                   list_for_integrate,
                   old_omega,
                   print_time_of_this_func, plot_gradien_loss,plot_consystency,print_num_of_restart_gradient,print_tmp_cons_and_loss):
    start_time_of_this_func = 0
    if print_time_of_this_func==True:
        start_time_of_this_func = time.time()
    l = 0
    while (True):
        z_list, train_info = train_and_save_z(f=f, num_of_rules=num_of_rules,dimension=dimension , train_dict=train_dict,init_z= init_z,distrib_of_rules=distrib_of_rules,
                                              min_cons=min_cons,
                                              min_norm=min_norm,
                                              min_distr=min_distr,
                                              previous_z_lists=previous_z_lists,
                                              problems_areas_info=problems_areas_info,
                                              list_for_integrate=list_for_integrate,
                                              old_omega=old_omega,
                                              check_target_values=check_target_values,
                                              plot_gradien_loss=plot_gradien_loss, a=a, h=h, coeff_list=coeff_list,print_tmp_cons_and_loss=print_tmp_cons_and_loss)
        l += 1
        # if train_info["last_consistency"] < 0.7 and train_info["last_norm"] < 0.001:
        if check_target_values==True:
            if train_info["last_consistency"] < min_cons and train_info["last_norm"] < min_norm and train_info["last_distr"]<min_distr:
                if (l > 1):
                    if print_num_of_restart_gradient == True:
                        print("\nлишних попыток на взятие градиентов {}".format(l - 1))
                if plot_consystency==True:
                    plot_consistency(z_list, rules, omega, a, h)
                break
        else:
            if plot_consystency == True:
                plot_consistency(z_list, rules, omega, a, h)
            break
    if print_time_of_this_func == True:
        print("\n     time of gradients: {} sek".format(time.time() - start_time_of_this_func))

    return z_list


def integrate_z_di(z, np_omega, new_omega_list, list_for_integrate, print_time_of_this_func,print_time_of_integration_for_each_rect,print_tmp_computed):

    start_time_of_integration = 0
    if print_time_of_this_func==True:
        start_time_of_integration = time.time()

    num_of_intersections_areas = 0
    if print_tmp_computed==True:
        for i in range(len(list_for_integrate)):
            if len(list_for_integrate[i][1]) > 1:
                num_of_intersections_areas += 1

    list_of_integrals_z_di = []
    tmp_computed = 0
    for i in range(len(list_for_integrate)):

        if len(list_for_integrate[i][1]) > 1:

            integral_in_rect = intergate_list_of_rules_on_tmp_new_omega(z, old_omega=np_omega, tmp_new_omega=new_omega_list[i],
                                                                        list_of_rules=list_for_integrate[i][1], print_time_of_integration_for_each_rect=print_time_of_integration_for_each_rect)
            list_of_integrals_z_di.append(integral_in_rect)
            if print_tmp_computed==True:
                tmp_computed+=1
                print("computed {} of {}".format(tmp_computed, num_of_intersections_areas))


    if print_time_of_this_func == True:
        print("\n     time of integration: {} sek".format(time.time() - start_time_of_integration))

    return list_of_integrals_z_di


def check_and_plot_p(name_of_gif_and_png,num_of_fig,np_z, np_omega, rules, list_of_integrals_z_di, list_for_integrate, new_omega_list):

    dfs = np.linspace(0, 6500, num_of_fig)
    # for i in range(100):
        # input = np.asarray([dfs[i], 50])
        # output = eval_list_of_integrals(input=input, old_z=np_z, old_omega=np_omega, list_for_integrate=list_for_integrate,
        #                                                          new_omega_list=new_omega_list, list_of_integrals_z_di=list_of_integrals_z_di)
        # print("Df:{: f} V:{: f} gas_brake:{: f}".format(input[0], input[1], output[0]))

        #  построим совместное распределение p(x_1, x_2, x_3) при фиксированном x_1 - Df
    print("making figures")
    for i in range(num_of_fig):
        print(i)
        plot_x_2_x_3_use_eval_list(name_of_gif_and_png,input=[dfs[i]], old_z=np_z, old_omega=np_omega, list_for_integrate=list_for_integrate,new_omega_list=new_omega_list, list_of_integrals_z_di=list_of_integrals_z_di, index=i)
    print("making figures done")

    # https://ezgif.com/gif-to-mp4/ezgif-1-0d8c4320a75d.gif  best site
    print("making gif")
    with imageio.get_writer("D:\saved_gifs\\" +name_of_gif_and_png+".gif", mode='I') as writer:

        for i in range(num_of_fig):
            print(i)
            image = imageio.imread("D:\\saved_fig\\"+name_of_gif_and_png+"_" + str(i) + ".png")
            writer.append_data(image)
    print("making gif done")
def check_and_plot_muptiple_p(np_z_list, np_omega, rules, list_of_integrals_z_di_list, list_for_integrate, new_omega_list):
    dfs = np.linspace(0, 6500, 200)



def loss_for_razb_potoch(trainable_razb, func_dict, a, b, N):
    loss = torch.zeros(1, requires_grad=False)  # интегральное расстояние

    for k in range((N + 1 - 2) - 1):
        for i in range(10):
            loss += (
                            torch.square(
                                compute_f_x(func_dict,
                                            trainable_razb[k] + i * (trainable_razb[k + 1] - trainable_razb[k]) / 10) -
                                compute_f_x(func_dict,
                                            trainable_razb[k] + (trainable_razb[k + 1] - trainable_razb[k]) / 2)
                            )
                            +
                            torch.square(
                                compute_f_x(func_dict, trainable_razb[k] + (i + 1) * (
                                        trainable_razb[k + 1] - trainable_razb[k]) / 10) -
                                compute_f_x(func_dict,
                                            trainable_razb[k] + (trainable_razb[k + 1] - trainable_razb[k]) / 2)
                            )
                    ) / 2 \
                    * (trainable_razb[k + 1] - trainable_razb[k]) / 10

    # еще нужно посчитать интеграл в первом и последнем отрезке, так как там крайние риски не варьируемые параметры
    for i in range(10):
        loss += (torch.square(
            compute_f_x(func_dict, a + i * (trainable_razb[0] - a) / 10) - compute_f_x(func_dict,
                                                                                       a + (trainable_razb[0] - a) / 2)) \
                 + torch.square(compute_f_x(func_dict, a + (i + 1) * (trainable_razb[0] - a) / 10
                                            ) - compute_f_x(func_dict,
                                                            a + (trainable_razb[0] - a) / 2))) / 2 \
                * (trainable_razb[0] - a) / 10

    for i in range(10):
        loss += (torch.square(
            compute_f_x(func_dict, trainable_razb[N - 2] + i * (b - trainable_razb[N - 2]) / 10) - compute_f_x(
                func_dict,
                trainable_razb[N - 2] + (b - trainable_razb[N - 2]) / 2)) \
                 + torch.square(
                    compute_f_x(func_dict, trainable_razb[N - 2] + (i + 1) * (b - trainable_razb[N - 2]) / 10
                                ) - compute_f_x(func_dict,
                                                trainable_razb[N - 2] + (b - trainable_razb[N - 2]) / 2))) / 2 \
                * (b - trainable_razb[N - 2]) / 10
    return loss


def learn_razb(func_dict, a, b, N):
    trainable_razb = torch.tensor(np.linspace(a, b, N + 1).tolist()[1:-1],
                                  requires_grad=True)  # a и b не модифицируем т.к. эти точки определяют границы области определения и за них вообще говоря выходить нельзя и сужаться относительно них тоже нельзя

    print(func_dict["name"])
    train_dict = {}
    if func_dict["name"] == "low_Df":
        train_dict = {
            "lr": 20,
            "num_of_epochs": 150

        }

    if func_dict["name"] == "medium_Df":
        train_dict = {
            "lr": 20,
            "num_of_epochs": 150,
            100: {"lr": 15}
        }
    if func_dict["name"] == "high_Df":
        train_dict = {
            "lr": 30,
            "num_of_epochs": 600,
            300: {"lr": 15},
            400: {"lr": 7.5}
        }
    if func_dict["name"] == "low_V":
        train_dict = {
            "lr": 0.2,
            "num_of_epochs": 22

        }
    if func_dict["name"] == "medium_V":
        train_dict = {
            "lr": 0.2,
            "num_of_epochs": 43

        }
    if func_dict["name"] == "high_V":
        train_dict = {
            "lr": 0.20,
            "num_of_epochs": 100,
            80: {"lr": 0.1}

        }
    if func_dict["name"] == "low_brake":
        train_dict = {
            "lr": 0.002,
            "num_of_epochs": 30,
            10: {"lr": 0.001}

        }
    if func_dict["name"] == "medium_brake":
        train_dict = {
            "lr": 0.01,
            "num_of_epochs": 30,
            20: {"lr": 0.005}

        }
    if func_dict["name"] == "high_brake":
        train_dict = {
            "lr": 0.01,
            "num_of_epochs": 40,
            20: {"lr": 0.005}

        }
    if func_dict["name"] == "low_gas":
        train_dict = {
            "lr": 0.002,
            "num_of_epochs": 30,
            10: {"lr": 0.001}

        }
    if func_dict["name"] == "medium_gas":
        train_dict = {
            "lr": 0.01,
            "num_of_epochs": 30,
            20: {"lr": 0.005}

        }
    if func_dict["name"] == "high_gas":
        train_dict = {
            "lr": 0.01,
            "num_of_epochs": 40,
            20: {"lr": 0.005}

        }

    optimizer = torch.optim.Adam([trainable_razb], train_dict["lr"], [0.5, 0.7])
    loss_vec = []
    for i in range(train_dict["num_of_epochs"]):

        if i in train_dict:
            for g in optimizer.param_groups:
                g['lr'] = train_dict[i]["lr"]
        optimizer.zero_grad()
        loss = loss_for_razb_potoch(trainable_razb, func_dict, a, b, N)
        print("\r>> {} loss: {}".format(i,
                                        loss.cpu().detach().numpy()), end='')
        # print(trainable_razb.detach().numpy())
        loss_vec.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()

        check_narushenie_arr = trainable_razb.detach().numpy()
        for i in range(len(check_narushenie_arr) - 1):
            if check_narushenie_arr[i] >= check_narushenie_arr[i + 1]:
                print("точки поменялись местати")
                time.sleep(100)

    fig_loss, axs_loss = plt.subplots(1, 1)
    loss_line, = axs_loss.plot(loss_vec)
    axs_loss.set_title("loss")
    axs_loss.set_yscale("linear")
    plt.show(block=False)
    return [a] + trainable_razb.detach().numpy().tolist() + [b]


def train_razb_potoch_approx(rules):
    unique_func_names = {}
    for rule in rules:
        for distr in rules[rule]:
            func_name = rules[rule][distr]["func"]["name"]
            if func_name not in unique_func_names.keys():
                unique_func_names.update({func_name: 0})
    func_razbs = {}

    for rule in rules:
        for distr in rules[rule]:
            a = rules[rule][distr]["a"]
            b = rules[rule][distr]["b"]
            N = rules[rule][distr]["N"]
            # построим разбиение
            if unique_func_names[rules[rule][distr]["func"]["name"]] == 0:
                func = rules[rule][distr]["func"]
                razb = learn_razb(func, a, b, N)
                func_razbs.update({rules[rule][distr]["func"]["name"]: razb})
                rules[rule][distr].update({"razb": razb})
                unique_func_names[rules[rule][distr]["func"]["name"]] = 1

                # посмотрим как изменилось разбиение после обучения
                fig_loss, axs_loss = plt.subplots(1, 1)
                arr_x_for_plot = np.linspace(a, b, 1000)
                arr_y_for_plot = np.zeros(1000, )
                for i in range(len(arr_y_for_plot)):
                    arr_y_for_plot[i] = compute_f_x(func, torch.tensor(arr_x_for_plot[i])).detach().numpy()
                init_func, = axs_loss.plot(arr_x_for_plot, arr_y_for_plot, label="init")

                # кусочно постоянная функция на равномерном разбиении
                np_linspace_for_init_razb = np.linspace(a, b, N + 1)
                arr_y_for_init_razb = np.zeros(1000, )
                for i in range(len(arr_y_for_init_razb)):
                    # нужно найти в какой кусок init_razb мы попали
                    index_of_otr_in_init_razb = -1
                    for j in range(len(np_linspace_for_init_razb) - 1):
                        if np_linspace_for_init_razb[j] <= arr_x_for_plot[i] <= np_linspace_for_init_razb[j + 1]:
                            index_of_otr_in_init_razb = j
                            break
                    arr_y_for_init_razb[i] = compute_f_x(func, torch.tensor(
                        np_linspace_for_init_razb[index_of_otr_in_init_razb] + (
                                np_linspace_for_init_razb[index_of_otr_in_init_razb + 1] -
                                np_linspace_for_init_razb[index_of_otr_in_init_razb]) / 2)).detach().numpy()
                init_razb_func, = axs_loss.plot(arr_x_for_plot, arr_y_for_init_razb, label="ravn_razb")
                # кусочно постоянная функция на неравномерном разбиении
                neravn_razb = rules[rule][distr]["razb"]
                arr_y_for_non_linear_razb = np.zeros(1000, )
                for i in range(len(arr_y_for_non_linear_razb)):
                    # нужно найти в какой кусок non_linear_razb мы попали
                    index_of_otr_in_non_linear_razb = -1
                    for j in range(len(neravn_razb) - 1):
                        if neravn_razb[j] <= arr_x_for_plot[i] <= neravn_razb[j + 1]:
                            index_of_otr_in_non_linear_razb = j
                            break
                    arr_y_for_non_linear_razb[i] = compute_f_x(func, torch.tensor(
                        neravn_razb[index_of_otr_in_non_linear_razb] + (
                                neravn_razb[index_of_otr_in_non_linear_razb + 1] - neravn_razb[
                            index_of_otr_in_non_linear_razb]) / 2)).detach().numpy()
                non_linear_razb_func, = axs_loss.plot(arr_x_for_plot, arr_y_for_non_linear_razb,
                                                      label="non_linear_razb")

                axs_loss.set_yscale("linear")
                axs_loss.legend()
                plt.show(block=False)
            else:
                rules[rule][distr].update({"razb": func_razbs[rules[rule][distr]["func"]["name"]]})

    filepath_with_rules = "C:\\writed_rules\\rules_with_non_linear_razb.txt"
    torch.save(rules, filepath_with_rules)
    return filepath_with_rules

def fill_ranv_razb(rules,fill_same_N = False,same_N=-1):
    if fill_same_N ==True:
        for rule in rules:
            for distr in rules[rule]:
                a = rules[rule][distr]["a"]
                b = rules[rule][distr]["b"]
                rules[rule][distr]["N"] = same_N
                rules[rule][distr]["razb"] = np.linspace(a, b, same_N + 1).tolist()
    else:
        for rule in rules:
            for distr in rules[rule]:
                a = rules[rule][distr]["a"]
                b = rules[rule][distr]["b"]
                rules[rule][distr]["razb"] = np.linspace(a, b, rules[rule][distr]["N"] + 1).tolist()
    return rules

def chat_with_sim(filename_for_messaging_with_engine,size_of_population,
                  list_with_z_np, old_omega, list_for_integrate,
                  new_omega_list, list_of_integrals_z_di):
    clear_dir_for_chat_with_sim(path_to_chating="C:\\for_engine_commands")
    file_for_messaging_with_engine = open(filename_for_messaging_with_engine, "a")
    file_for_messaging_with_engine.write(str(size_of_population))
    file_for_messaging_with_engine.close()
    print("wait ready sim")
    while True:
        if os.path.exists("C:\\for_engine_commands\\about_ready_env.txt"):
            os.remove("C:\\for_engine_commands\\about_ready_env.txt")
            break

    obs_path = []
    act_path = []
    ready_to_read_paths = []
    python_recorded_the_actions_paths = []
    agent_recorded_the_observations_paths = []
    paths_with_end_of_sim_info_for_agents = []
    for i in range(size_of_population):
        obs_path.append("C:\\for_engine_commands\\obs_car_" + str(i + 1) + ".txt")
        act_path.append("C:\\for_engine_commands\\act_car_" + str(i + 1) + ".txt")
        ready_to_read_paths.append("C:\\for_engine_commands\\python_ready_to_read_" + str(i + 1) + ".txt")
        python_recorded_the_actions_paths.append(
            "C:\\for_engine_commands\\python_recorded_the_actions_" + str(i + 1) + ".txt")
        agent_recorded_the_observations_paths.append(
            "C:\\for_engine_commands\\agent_recorded_the_observations_" + str(i + 1) + ".txt")
        paths_with_end_of_sim_info_for_agents.append(
            "C:\\for_engine_commands\\EndOfSimInfoForCar_" + str(i + 1) + ".txt")


    info_from_simulation = {}
    obs = np.zeros(2, )

    print("start messaging")
    for i in range(size_of_population):
        ready_to_read_file = open(ready_to_read_paths[i], "w")
        ready_to_read_file.close()
    tmp_cars = np.ones(size_of_population, )

    need_get_answer = 0
    while True:

        # для тех машин, сообщение от которых мы приняли и отправили ответ, нужно снова отправить сообщение, что мы готовы читать наблюдения
        for i in range(size_of_population):
            if tmp_cars[i] == 0:
                ready_to_read_file = open(ready_to_read_paths[i], "w")
                ready_to_read_file.close()
                tmp_cars[i] = 1

        if os.path.exists("C:\\for_engine_commands\\about_end_of_sim.txt"):
            # here need read sim info
            for i in range(size_of_population):
                with open(paths_with_end_of_sim_info_for_agents[i], "r") as f:
                    mode = int(f.readline())
                    value = float(f.readline())
                    info_from_simulation.update({i: [mode, value]})
            os.remove("C:\\for_engine_commands\\about_end_of_sim.txt")
            break

        for i in range(size_of_population):
            if os.path.exists(agent_recorded_the_observations_paths[i]):
                with open(agent_recorded_the_observations_paths[i], 'w') as f_agent_recorded_obs:
                    # отвечаем на наблюдения для конкретной машинки и удаляем флаг "готов читать" пока вычисляем ответ для
                    # для этой же конкретной машинки
                    os.remove(ready_to_read_paths[i])
                    # os.remove(agent_recorded_the_observations_paths[i])
                    tmp_cars[i] = 0  # запоминаем что сообщение от конкретной машинки мы прочили, пока не существует файла
                    # ready_to_read машинка туда не будет писать наблюдения
                    need_get_answer = 1
                    with open(obs_path[i], "r") as f:
                        try:
                            obs[0] = f.readline()
                            obs[1] = f.readline()
                        except:
                            print("empty file with obs {}".format(obs_path[i]))
                            need_get_answer = 0
                    if (need_get_answer == 1):
                        if obs[0] > 6500.0:
                            obs[0] = 6500.0
                        if obs[0] < 0.0:
                            obs[0] = 0.0

                        if obs[1] > 94.0:
                            obs[1] = 94.0
                        if obs[1] < 0.0:
                            obs[1] = 0.0
                        output = eval_list_of_integrals(input=obs, old_z=list_with_z_np[i], old_omega=old_omega,
                                                        list_for_integrate=list_for_integrate,
                                                        new_omega_list=new_omega_list, list_of_integrals_z_di=list_of_integrals_z_di[i])
                        np.savetxt(act_path[i], output)
                        obs[0] = 0
                        obs[1] = 0
                        python_recorded_the_action_file = open(python_recorded_the_actions_paths[i], "w")
                        python_recorded_the_action_file.close()
                try:
                    os.remove(agent_recorded_the_observations_paths[i])
                except:
                    print("can't remove file {}".format(agent_recorded_the_observations_paths[i]))
    return info_from_simulation

def take_multiple_gradients_from_one_init_point(
            num_of_rules,
            dimension,
            rules,
            omega,
            train_dicts,
            a, h, f, coeff_list,
            init_z,distrib_of_rules,
            previous_z_lists,
            print_time_of_this_func,
            plot_gradien_loss,
            plot_consystency,
            print_num_of_restart_gradient,
            print_tmp_cons_and_loss):
    z_lists = []
    num_of_solutions = len(train_dicts)
    for i in range(num_of_solutions):
        z_list = take_gradients(
            num_of_rules,
            dimension,
            rules,
            omega,
            train_dicts[i],
            a, h, f, coeff_list,
            init_z,distrib_of_rules,
            previous_z_lists,
            print_time_of_this_func,
            plot_gradien_loss,
            plot_consystency,
            print_num_of_restart_gradient,
            print_tmp_cons_and_loss)
        z = torch_to_numpy_z(z_list)
        z_lists.append(z)
    return z_lists

# def plot_debug_distib():
#
#     check_and_plot_p(np_z=z, np_omega=np_omega,rules=rules,list_of_integrals_z_di=list_of_integrals_z_di, list_for_integrate=list_for_integrate, new_omega_list=new_omega_list)
#
#
#     obs = np.asarray([3000, 40])
#     act_arr = eval_list_of_integrals(input=obs, old_z=z, old_omega=omega, list_for_integrate=list_for_integrate,
#                                      new_omega_list=new_omega_list, list_of_integrals_z_di=list_of_integrals_z_di)
#     print(act_arr)

def chat_with_sim_one_contoller(
                                filename_for_messaging_with_engine,
                                num_of_duplicates,
                                z_np,
                                old_omega,
                                list_for_integrate,
                                new_omega,
                                integral_z_di):
    clear_dir_for_chat_with_sim(path_to_chating="C:\\for_engine_commands")
    file_for_messaging_with_engine = open(filename_for_messaging_with_engine, "a")
    file_for_messaging_with_engine.write(str(num_of_duplicates))
    file_for_messaging_with_engine.close()
    print("wait ready sim")
    while True:
        if os.path.exists("C:\\for_engine_commands\\about_ready_env.txt"):
            os.remove("C:\\for_engine_commands\\about_ready_env.txt")
            break

    obs_path = []
    act_path = []
    ready_to_read_paths = []
    python_recorded_the_actions_paths = []
    agent_recorded_the_observations_paths = []
    paths_with_end_of_sim_info_for_agents = []
    for i in range(num_of_duplicates):
        obs_path.append("C:\\for_engine_commands\\obs_car_" + str(i + 1) + ".txt")
        act_path.append("C:\\for_engine_commands\\act_car_" + str(i + 1) + ".txt")
        ready_to_read_paths.append("C:\\for_engine_commands\\python_ready_to_read_" + str(i + 1) + ".txt")
        python_recorded_the_actions_paths.append(
            "C:\\for_engine_commands\\python_recorded_the_actions_" + str(i + 1) + ".txt")
        agent_recorded_the_observations_paths.append(
            "C:\\for_engine_commands\\agent_recorded_the_observations_" + str(i + 1) + ".txt")
        paths_with_end_of_sim_info_for_agents.append(
            "C:\\for_engine_commands\\EndOfSimInfoForCar_" + str(i + 1) + ".txt")


    info_from_simulation = {}
    obs = np.zeros(2, )

    print("start messaging")
    for i in range(num_of_duplicates):
        ready_to_read_file = open(ready_to_read_paths[i], "w")
        ready_to_read_file.close()
    tmp_cars = np.ones(num_of_duplicates, )

    need_get_answer = 0
    while True:

        # для тех машин, сообщение от которых мы приняли и отправили ответ, нужно снова отправить сообщение, что мы готовы читать наблюдения
        for i in range(num_of_duplicates):
            if tmp_cars[i] == 0:
                ready_to_read_file = open(ready_to_read_paths[i], "w")
                ready_to_read_file.close()
                tmp_cars[i] = 1

        if os.path.exists("C:\\for_engine_commands\\about_end_of_sim.txt"):
            # here need read sim info
            for i in range(num_of_duplicates):
                with open(paths_with_end_of_sim_info_for_agents[i], "r") as f:
                    mode = int(f.readline())
                    value = float(f.readline())
                    info_from_simulation.update({i: [mode, value]})
            os.remove("C:\\for_engine_commands\\about_end_of_sim.txt")
            break

        for i in range(num_of_duplicates):
            if os.path.exists(agent_recorded_the_observations_paths[i]):
                with open(agent_recorded_the_observations_paths[i], 'w') as f_agent_recorded_obs:
                    # отвечаем на наблюдения для конкретной машинки и удаляем флаг "готов читать" пока вычисляем ответ для
                    # для этой же конкретной машинки
                    os.remove(ready_to_read_paths[i])
                    # os.remove(agent_recorded_the_observations_paths[i])
                    tmp_cars[i] = 0  # запоминаем что сообщение от конкретной машинки мы прочили, пока не существует файла
                    # ready_to_read машинка туда не будет писать наблюдения
                    need_get_answer = 1
                    with open(obs_path[i], "r") as f:
                        try:
                            obs[0] = f.readline()
                            obs[1] = f.readline()
                        except:
                            print("empty file with obs {}".format(obs_path[i]))
                            need_get_answer = 0
                    if (need_get_answer == 1):
                        if obs[0] > 6500.0:
                            obs[0] = 6500.0
                        if obs[0] < 0.0:
                            obs[0] = 0.0

                        if obs[1] > 94.0:
                            obs[1] = 94.0
                        if obs[1] < 0.0:
                            obs[1] = 0.0
                        output = eval_list_of_integrals(input=obs, old_z=z_np, old_omega=old_omega,
                                                        list_for_integrate=list_for_integrate,
                                                        new_omega_list=new_omega, list_of_integrals_z_di=integral_z_di)
                        np.savetxt(act_path[i], output)
                        obs[0] = 0
                        obs[1] = 0
                        python_recorded_the_action_file = open(python_recorded_the_actions_paths[i], "w")
                        python_recorded_the_action_file.close()
                try:
                    os.remove(agent_recorded_the_observations_paths[i])
                except:
                    print("can't remove file {}".format(agent_recorded_the_observations_paths[i]))
    return info_from_simulation