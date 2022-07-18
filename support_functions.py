import os
import shutil

import imageio
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import pprint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from scipy.integrate import quad
from rules import integrand_FI

from make_random_func import compute_seq_vec
from make_random_func import compute_gauss


def compute_f(func, segment):
    """

    :param func: {"name": "FI", "params": [0, 1]}
    :param segment: torch.zeros(size=(2,), requires_grad=False) -левый и правый конец отрезка
    :return: значение функции на середине отрезка
    """
    if func["name"] == "low_Df":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (- func["params"][0] + segment[0] + (segment[1] - segment[0]) / 2) / func["params"][1]))
    if func["name"] == "medium_Df":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + segment[0] + (segment[1] - segment[0]) / 2) / func["params"][1]))
    if func["name"] == "high_Df":
        return 1 / func["norm"] * 1 / (1 + torch.exp(
            -func["params"][1] * (segment[0] + (segment[1] - segment[0]) / 2 - func["params"][0])))

    if func["name"] == "low_V":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + segment[0] + (segment[1] - segment[0]) / 2) / func["params"][1]))
    if func["name"] == "medium_V":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + segment[0] + (segment[1] - segment[0]) / 2) / func["params"][1]))
    if func["name"] == "high_V":
        return 1 / func["norm"] * 1 / (1 + torch.exp(
            -func["params"][1] * (segment[0] + (segment[1] - segment[0]) / 2 - func["params"][0])))

    if func["name"] == "default_brake":
        return torch.ones(1) / 0.003
    if func["name"] == "low_brake":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + segment[0] + (segment[1] - segment[0]) / 2) / func["params"][1]))
    if func["name"] == "medium_brake":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + segment[0] + (segment[1] - segment[0]) / 2) / func["params"][1]))
    if func["name"] == "high_brake":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + segment[0] + (segment[1] - segment[0]) / 2) / func["params"][1]))

    if func["name"] == "default_gas":
        return torch.ones(1) / 0.003
    if func["name"] == "low_gas":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + segment[0] + (segment[1] - segment[0]) / 2) / func["params"][1]))
    if func["name"] == "medium_gas":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + segment[0] + (segment[1] - segment[0]) / 2) / func["params"][1]))
    if func["name"] == "high_gas":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + segment[0] + (segment[1] - segment[0]) / 2) / func["params"][1]))
    if func["name"] == "FI":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + segment[0] + (segment[1] - segment[0]) / 2) / func["params"][1]))

def compute_f_x(func, x):
    """

    :param func: {"name": "FI", "params": [0, 1]}
    :param segment: torch.zeros(size=(2,), requires_grad=False) -левый и правый конец отрезка
    :return: значение функции на середине отрезка
    """
    if func["name"] == "low_Df":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (- func["params"][0] + x) / func["params"][1]))
    if func["name"] == "medium_Df":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + x) / func["params"][1]))
    if func["name"] == "high_Df":
        return 1 / func["norm"] * 1 / (1 + torch.exp(
            -func["params"][1] * (x - func["params"][0])))

    if func["name"] == "low_V":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + x) / func["params"][1]))
    if func["name"] == "medium_V":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + x) / func["params"][1]))
    if func["name"] == "high_V":
        return 1 / func["norm"] * 1 / (1 + torch.exp(
            -func["params"][1] * (x - func["params"][0])))

    if func["name"] == "default_brake":
        return torch.ones(1) / 0.003
    if func["name"] == "low_brake":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + x) / func["params"][1]))
    if func["name"] == "medium_brake":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + x) / func["params"][1]))
    if func["name"] == "high_brake":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + x) / func["params"][1]))

    if func["name"] == "default_gas":
        return torch.ones(1) / 0.003
    if func["name"] == "low_gas":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + x) / func["params"][1]))
    if func["name"] == "medium_gas":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + x) / func["params"][1]))
    if func["name"] == "high_gas":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + x) / func["params"][1]))
    if func["name"] == "FI":
        return 1 / func["norm"] * torch.exp(
            -1 / 2 * torch.square(
                (-func["params"][0] + x) / func["params"][1]))


def plot_f(axs, distribution, z, number_of_rule, number_of_distribution, omega, rules, a, h):
    axs.grid(True, 'both', 'both')
    x = np.linspace(distribution["a"], distribution["b"], 100)
    x_plot = np.zeros(99, )
    y = np.zeros(99, )
    for i in range(100 - 1):
        y[i] = compute_f(distribution["func"], torch.tensor([x[i], x[i + 1]])).cpu().detach().numpy()
        x_plot[i] = x[i] + (x[i + 1] - x[i]) / 2
    line_1, = axs.plot(x_plot, y)
    line_1.set_label('ground truth')

    sum_tuple = []
    for tmp_i in range(len(omega[number_of_rule - 1])):
        if (tmp_i != (number_of_distribution - 1)):
            sum_tuple.append(tmp_i)
    sum_tuple = tuple(sum_tuple)
    y2 = (((torch.sum(z[number_of_rule - 1] * a[number_of_rule - 1], sum_tuple)) / (
        torch.sum(z[number_of_rule - 1] * a[number_of_rule - 1]))) / h[number_of_rule - 1][
              number_of_distribution - 1]).cpu().detach().numpy()
    tmp_x2 = omega[number_of_rule - 1][number_of_distribution - 1]
    x2 = torch.zeros(len(tmp_x2), )
    for i in range(len(x2)):
        x2[i] = (tmp_x2[i][0] + (tmp_x2[i][1] - tmp_x2[i][0]) / 2)
    x2 = x2.cpu().detach().numpy()
    line_2, = axs.plot(x2, y2, '*')
    line_2.set_label('approx')
    axs.legend()
    axs.set_title("rule: {} distr:{} {}".format(number_of_rule, number_of_distribution,
                                                rules[number_of_rule][number_of_distribution]['label']))
    if ("default" in distribution["func"]["name"]):
        axs.set_yscale("linear")
    else:
        axs.set_yscale("linear")


def my_loss(z, f, num_of_rules, a, h, coeff_list, init_z, distrib_of_rules,
            index_of_iteration,index_of_apply_raznost,max_squares,
            indicies_of_target_problem_areas_k_rules_indicies_of_space):
    z_list = []
    for i in range(num_of_rules):
        z_list.append(torch.exp(z[i]))

    z_a_list = []
    for i in range(num_of_rules):
        z_a_list.append(z_list[i] * a[i])

    D_i_vec = torch.zeros(num_of_rules, requires_grad=False)
    for i in range(num_of_rules):
        D_i_vec[i] = torch.sum(z_a_list[i])

    # норма совметсного распредления
    norm_loss = torch.square(1 - torch.sum(D_i_vec))
    distrib_of_rules_loss = torch.sum(torch.square(D_i_vec - distrib_of_rules))

    S_p_j_k = []
    consistency_loss = torch.zeros(1, requires_grad=False)

    for i in range(num_of_rules):
        S_p_j_k.append([])
        S_p_j_k[i].append((torch.sum(z_a_list[i], (1, 2)) / h[i][0]) / D_i_vec[i])
        S_p_j_k[i].append((torch.sum(z_a_list[i], (0, 2)) / h[i][1]) / D_i_vec[i])
        S_p_j_k[i].append((torch.sum(z_a_list[i], (0, 1)) / h[i][2]) / D_i_vec[i])

        # consistency_loss = consistency_loss + \
        #                    coeff_list[i][0] * torch.sum(
        #     torch.square((f[i][0] - (torch.sum(z_a_list[i], (1, 2)) / h[i][0]) / D_i_vec[i]))) + \
        #                    coeff_list[i][1] * torch.sum(
        #     torch.square((f[i][1] - (torch.sum(z_a_list[i], (0, 2)) / h[i][1]) / D_i_vec[i]))) + \
        #                    coeff_list[i][2] * torch.sum(
        #     torch.square((f[i][2] - (torch.sum(z_a_list[i], (0, 1)) / h[i][2]) / D_i_vec[i])))

    ro_vec = torch.zeros(num_of_rules, requires_grad=False)

    for i in range(num_of_rules):
        ro_vec[i] = torch.sum(a[i] * torch.square(init_z[i] - z_list[i]))
    ro = torch.sum(ro_vec)
    if index_of_iteration > index_of_apply_raznost:
        for i in range(num_of_rules):
            consistency_loss = consistency_loss + \
                               torch.sum(torch.square((f[i][0] - S_p_j_k[i][0]) / max_squares[i][0])) + \
                               torch.sum(torch.square((f[i][1] - S_p_j_k[i][1]) / max_squares[i][1])) + \
                               torch.sum(torch.square((f[i][2] - S_p_j_k[i][2]) / max_squares[i][2]))
        loss = consistency_loss + norm_loss + distrib_of_rules_loss + ro
        return loss, consistency_loss, norm_loss, ro, ro_vec, distrib_of_rules_loss
    else:
        for i in range(num_of_rules):
            consistency_loss = consistency_loss + \
            coeff_list[i][0] * torch.sum(torch.square(f[i][0] - S_p_j_k[i][0])) + \
            coeff_list[i][1] * torch.sum(torch.square(f[i][1] - S_p_j_k[i][1])) + \
            coeff_list[i][2] * torch.sum(torch.square(f[i][2] - S_p_j_k[i][2]))


    loss = consistency_loss + norm_loss + distrib_of_rules_loss + ro
    return loss, consistency_loss, norm_loss, ro, ro_vec, distrib_of_rules_loss


def train_and_save_z(f, num_of_rules, dimension, train_dict, init_z, distrib_of_rules,
                     min_cons,
                     min_norm,
                     min_distr,
                     previous_z_lists,
                     problems_areas_info,
                     list_for_integrate,
                     old_omega,
                     check_target_values,
                     plot_gradien_loss, a, h,
                     coeff_list,
                     print_tmp_cons_and_loss,
                     ):
    train_info = {}

    z = []

    distrib_of_rules_torch = torch.tensor(distrib_of_rules)

    for i in range(num_of_rules):
        # size_of_pth_tensor = []
        # for distribution in rules[rule]:
        #     size_of_pth_tensor.append(rules[rule][distribution]["N"])
        # z_np = -2 + 4*torch.rand(size=tuple(size_of_pth_tensor))
        # z.append(z_np.clone().detach().requires_grad_(True))

        # here make initial point
        z.append(torch.log(init_z[i]).clone().detach().requires_grad_(True))

    previous_z_lists_consts = []
    for i in range(len(previous_z_lists)):
        previous_z_lists_consts.append([])
        for j in range(len(previous_z_lists[i])):
            previous_z_lists_consts[i].append(torch.from_numpy(previous_z_lists[i][j]).requires_grad_(False))

    optimizer = torch.optim.Adam(z, train_dict["lr"], [0.5, 0.7])

    loss_vec = np.zeros(train_dict["max_num_of_epochs"], )
    consyst_vec = np.zeros(train_dict["max_num_of_epochs"], )
    norm_vec = np.zeros(train_dict["max_num_of_epochs"], )

    ro_vec = np.zeros(train_dict["max_num_of_epochs"], )
    distr_of_rules_vec = np.zeros(train_dict["max_num_of_epochs"], )
    reg_vec =np.zeros(train_dict["max_num_of_epochs"], )

    last_lr = train_dict["lr"]

    # max_squares
    max_squares= []
    for i in range(num_of_rules):
        max_squares.append([])
        for j in range(dimension):
            max= torch.square(torch.max(f[i][j]))
            min =torch.square(torch.min(f[i][j]))
            value_to_vzvesh = (max-min)/20
            max_squares[i].append(torch.square(value_to_vzvesh))

    last_index_for_plot = 0
    max_index = 5
    index_of_mean_computation = 5

    N_half= 1500

    max_problem = np.quantile(problems_areas_info,0.25)
    target_areas = []
    for i in range(len(list_for_integrate)):
        if problems_areas_info[i] <= max_problem:
            # rect = list_for_integrate[i][0]
            # indicies_of_rules= list_for_integrate[i][1]
            target_areas.append(list_for_integrate[i])
    indicies_of_target_problem_areas_k_rules_indicies_of_space = []
    for i in range(num_of_rules):
        indicies_of_target_problem_areas_k_rules_indicies_of_space.append([])

    for i in range(len(target_areas)):
        rect = target_areas[i][0]
        a_x=rect[0][0]
        b_x=rect[0][1]
        a_y=rect[1][0]
        b_y=rect[1][1]
        a_z=rect[2][0]
        b_z=rect[2][1]
        indicies_of_rules = target_areas[i][1]
        # print(rect)
        # print(indicies_of_rules)
        for index_of_rule in indicies_of_rules:
            from_indicies = get_indicies_in_old_omega([a_x,a_y,a_z],old_omega,index_of_rule)
            to_indicies = get_indicies_in_old_omega([b_x,b_y,b_z],old_omega,index_of_rule)
            indicies_of_target_problem_areas_k_rules_indicies_of_space[index_of_rule].append([from_indicies, to_indicies])
            # print(indicies_of_target_problem_areas_k_rules_indicies_of_space[index_of_rule])

    for i in range(train_dict["max_num_of_epochs"]):
        if i in train_dict:
            for g in optimizer.param_groups:
                g['lr'] = train_dict[i]["lr"]
                last_lr = train_dict[i]["lr"]

        if i > max_index:
            last_mean_loss = np.mean(loss_vec[i - index_of_mean_computation:i])
            last_loss = loss_vec[i - 1]

            if last_loss < last_mean_loss*0.9:
                last_lr = last_lr*0.99
                for g in optimizer.param_groups:
                    g['lr'] = last_lr


        optimizer.zero_grad()
        loss, consistency, norm, ro, ro_rules_values, distr_of_rules = my_loss(z, f, num_of_rules, a, h,
                                                                               coeff_list,
                                                                               init_z=init_z,
                                                                               distrib_of_rules=distrib_of_rules_torch,
                                                                               index_of_iteration=i,
                                                                               index_of_apply_raznost=N_half,max_squares=max_squares,
                                                                               indicies_of_target_problem_areas_k_rules_indicies_of_space=indicies_of_target_problem_areas_k_rules_indicies_of_space
                                                                               )
        norm_loss = float(norm.cpu().detach().numpy())
        cons_loss = float(consistency.cpu().detach().numpy())
        ro_loss = float(ro.cpu().detach().numpy())
        distr_of_rules_loss = float(distr_of_rules.cpu().detach().numpy())
        # reg_loss= float(reg_torch.cpu().detach().numpy())



        if print_tmp_cons_and_loss == True:
            print(
                "\r>>   {} ep lr {:10.11f} consistency: {:10.11f}   norm: {:10.11f}  ro {:10.11f} distr {:10.11f}".format(
                    i,last_lr,
                    cons_loss,
                    norm_loss,
                    ro_loss, distr_of_rules_loss
                    ),
                end='')
        loss_for_plot = float(loss.cpu().detach().numpy())

        loss_vec[i] = loss_for_plot
        consyst_vec[i] = cons_loss
        norm_vec[i] = norm_loss

        ro_vec[i] = ro_loss
        distr_of_rules_vec[i] = distr_of_rules_loss
        # reg_vec[i]= reg_loss

        # if (cons_loss < 0.005 and norm_loss < 0.001 and distr_of_rules_loss < 0.001):
        if check_target_values==True:
            if (cons_loss < min_cons and norm_loss < min_norm and distr_of_rules_loss<min_distr):
                last_index_for_plot = i
                break
        loss.backward()
        optimizer.step()

    for i in range(len(z)):
        z[i] = torch.exp(z[i])

    if plot_gradien_loss == True:
        fig_loss, axs_loss = plt.subplots(1, 4)
        if last_index_for_plot == 0:
            last_index_for_plot = train_dict["max_num_of_epochs"] - 1
        # loss_line, = axs_loss[0].plot(loss_vec[:last_index_for_plot])
        # axs_loss[0].set_title("loss")
        # axs_loss[0].set_yscale("log")
        consistency_line, = axs_loss[0].plot(consyst_vec[:last_index_for_plot])
        axs_loss[0].set_title("consistency")
        axs_loss[0].set_yscale("linear")

        norm_line, = axs_loss[1].plot(norm_vec[:last_index_for_plot])
        axs_loss[1].set_title("norm")
        axs_loss[1].set_yscale("linear")

        ro_line, = axs_loss[2].plot(ro_vec[:last_index_for_plot])
        axs_loss[2].set_title("ro")
        axs_loss[2].set_yscale("linear")

        distr_line = axs_loss[3].plot(distr_of_rules_vec[:last_index_for_plot])
        axs_loss[3].set_title("distr")
        axs_loss[3].set_yscale("linear")

        # reg_line = axs_loss[4].plot(reg_vec[:last_index_for_plot])
        # axs_loss[4].set_title("reg in problems areas")
        # axs_loss[4].set_yscale("linear")

        plt.show(block=True)

    train_info.update({"last_consistency": consyst_vec[-1]})
    train_info.update({"last_norm": norm_vec[-1]})
    train_info.update({"last_distr":distr_of_rules_vec[-1]})
    return z, train_info


def plot_consistency(z, rules, omega, a, h):
    # print("########################################################################")
    print("drawing consistency:")
    print("     minimum and maximum values:")
    for p in range(len(z)):
        print("     rule number: {}, min: {}   max: {}".format(p + 1, torch.min(z[p]).cpu().detach().numpy(),
                                                               torch.max(z[p]).cpu().detach().numpy()))
    # print("########################################################################")
    figures = []
    axes = []
    for rule in rules:
        fig, axs = plt.subplots(1, len(z[rule - 1].size()))
        figures.append(fig)
        axes.append(axs)
        for distribution in rules[rule]:
            plot_f(axs[distribution - 1], rules[rule][distribution], z, rule, distribution, omega, rules, a, h)
    plt.show(block=True)


def compute_z(z, omega, list_of_args, p):
    """

    :param z:list of torch tensors
    :param omega: list of list of torch tensors
    :param list_of_args: x_1, x_2,...,x_8; x_1 in |R, x_2 in |R,...,x_8 in |R
    :param p: номер правила p=0,1,2,...80 например
    :return: z(x_1,x_2,...,x_8,i)
    """

    # нужно проверить не находится ли точка (x_1,x_2,...,x_8) вне прямоугольника
    for i in range(len(list_of_args)):
        if (list_of_args[i] < omega[p][i][0][0]) or (list_of_args[i] > omega[p][i][-1][1]):
            return 0.0

    list_of_indicies = []
    for i in range(len(list_of_args)):
        for k in range(len(omega[p][i])):
            if (list_of_args[i] >= omega[p][i][k][0]) and (list_of_args[i] <= omega[p][i][k][1]):
                list_of_indicies.append(k)
                break
    list_of_indicies = tuple(list_of_indicies)
    return z[p][list_of_indicies]


def make_list_of_rect_from_rules(rules):
    # list_of_rect = [
    #     [[ax1_1, bx1_1], [ax1_2, bx1_2]],
    #     [[ax2_1, bx2_1], [ax2_2, bx2_2]],
    #     [[ax3_1, bx3_1], [ax3_2, bx3_2]],
    #     [[ax4_1, bx4_1], [ax4_2, bx4_2]],
    # ]
    list_of_rect = []
    for rule in rules:
        list_of_rect.append([])
        for distribution in rules[rule]:
            list_of_rect[rule - 1].append([])
            list_of_rect[rule - 1][distribution - 1].append(rules[rule][distribution]["a"])
            list_of_rect[rule - 1][distribution - 1].append(rules[rule][distribution]["b"])
    return list_of_rect


def make_tensor_from_list_of_rects(list_of_rect):
    # функция не тестировалась
    axis_left_sides = []
    axis_right_sides = []
    for axis in range(len(list_of_rect[0])):
        axis_left_sides.append([])
        axis_right_sides.append([])
        for rect_index in range(len(list_of_rect)):
            axis_left_sides[axis].append(list_of_rect[rect_index][axis][0])
            axis_right_sides[axis].append(list_of_rect[rect_index][axis][1])
    axis_left_sides = np.asarray(axis_left_sides)
    axis_right_sides = np.asarray(axis_right_sides)
    mins_of_x_i = np.zeros(len(axis_left_sides), )
    maxs_of_x_i = np.zeros(len(axis_right_sides), )
    for axis in range(len(axis_left_sides)):
        mins_of_x_i[axis] = np.min(axis_left_sides[axis])
        maxs_of_x_i[axis] = np.max(axis_right_sides[axis])

    omega_for_rules = []
    for i in range(len(list_of_rect[0])):
        omega_for_rules.append([mins_of_x_i[i]])
    for i in range(len(list_of_rect[0])):
        while True:
            tmp_left_border = omega_for_rules[i][-1]
            if tmp_left_border == maxs_of_x_i[i]:
                break
            tmp_min_new_left_border = []
            args_of_tmp_min_new_left_border = []
            for p in range(len(list_of_rect)):
                if tmp_left_border > list_of_rect[p][i][1]:
                    continue
                for k in range(len(list_of_rect[p][i])):
                    if list_of_rect[p][i][k] > tmp_left_border:
                        tmp_min_new_left_border.append(list_of_rect[p][i][k])
                        break
            tmp = np.asarray(tmp_min_new_left_border)
            tmp_min = np.min(tmp)
            omega_for_rules[i].append(tmp_min)
    return_omega_for_rules = []
    for i in range(len(mins_of_x_i)):
        return_omega_for_rules.append([])
        for j in range(len(omega_for_rules[i]) - 1):
            return_omega_for_rules[i].append([omega_for_rules[i][j], omega_for_rules[i][j + 1]])

    # хардкод можно переписать с помощью рекурсии
    # инициализация
    tensor_with_information = []
    for i_1 in range(len(return_omega_for_rules[0])):
        tensor_with_information.append([])
        for i_2 in range(len(return_omega_for_rules[1])):
            tensor_with_information[i_1].append([])
            for i_3 in range(len(return_omega_for_rules[2])):
                tensor_with_information[i_1][i_2].append([])
    # заполнение информацией о накрытии каким-то(ми-то) правилом текущего i_1 i_2 i_3 i_4 -го прямоугольника
    for i_1 in range(len(return_omega_for_rules[0])):
        for i_2 in range(len(return_omega_for_rules[1])):
            for i_3 in range(len(return_omega_for_rules[2])):
                segment_0 = return_omega_for_rules[0][i_1]
                segment_1 = return_omega_for_rules[1][i_2]
                segment_2 = return_omega_for_rules[2][i_3]
                for index_of_rect in range(len(list_of_rect)):
                    if segment_0[0] >= list_of_rect[index_of_rect][0][0] and segment_0[1] <= \
                            list_of_rect[index_of_rect][0][1]:
                        if segment_1[0] >= list_of_rect[index_of_rect][1][0] and segment_1[1] <= \
                                list_of_rect[index_of_rect][1][1]:
                            if segment_2[0] >= list_of_rect[index_of_rect][2][0] and segment_2[1] <= \
                                    list_of_rect[index_of_rect][2][1]:
                                tensor_with_information[i_1][i_2][i_3].append(index_of_rect)

    return tensor_with_information, return_omega_for_rules


def torch_to_numpy_z(z):
    for i in range(len(z)):
        z[i] = z[i].cpu().detach().numpy()
    return z


def make_new_omega_for_rect(old_omega, rect, list_of_rules):
    # создадим неравномерную сетку по осям основываясь на предыдущем рабиении и границах области определения переменных
    # находим левую и правую границу по каждой оси

    new_omega_for_rect = []
    for i in range(len(rect)):
        new_omega_for_rect.append([rect[i][0]])
    #    зафикисировали переменную для которой будем находить риски водль координатной прямой
    for i in range(len(rect)):
        while True:
            tmp_left_border = new_omega_for_rect[i][-1]
            if tmp_left_border == rect[i][1]:
                break
            # проходимся только по тем праавилам, которые указаны в спец. списке, т.к. этот прямоугольник только
            # они и накрывают
            tmp_min_new_left_border = []
            for p in list_of_rules:
                # узнаем диапазон индексов по которым нужно передвигаться в p-м правиле
                # вдоль i-го направления
                j_min = 0
                j_max = 0
                if len(old_omega[p][i]) == 1:
                    j_min = 0
                    j_max = 1
                else:
                    for i_of_segment in range(len(old_omega[p][i])):
                        # если риска попала в один из отрезков разбиения - запомним это
                        if old_omega[p][i][i_of_segment][0] <= rect[i][0] <= old_omega[p][i][i_of_segment][1]:
                            j_min = i_of_segment
                            for i_of_following_segment in range(i_of_segment, len(old_omega[p][i])):
                                if old_omega[p][i][i_of_following_segment][0] <= rect[i][1] <= \
                                        old_omega[p][i][i_of_following_segment][1]:
                                    j_max = i_of_following_segment
                                    break
                            break
                # зная диапазон индексов вычислим риску следущую за текущей риской, ближайщую к текущей
                if len(old_omega[p][i]) == 1:
                    if old_omega[p][i][0][1] > tmp_left_border:
                        tmp_min_new_left_border.append(old_omega[p][i][0][1])
                else:
                    # так и не понял, почему добовление j_min, j_max + 1 в место len(old_omega[p][i] меняет решение а не
                    # только ускоряет вычисления
                    for i_of_segment in range(j_min, j_max + 1):
                        if old_omega[p][i][i_of_segment][0] > tmp_left_border:
                            tmp_min_new_left_border.append(old_omega[p][i][i_of_segment][0])
                            break
            if len(tmp_min_new_left_border) == 0:
                tmp_min_new_left_border.append(rect[i][1])
            tmp = np.asarray(tmp_min_new_left_border)
            new_omega_for_rect[i].append(np.min(tmp))

    return_new_omega = []
    for i in range(len(rect)):
        return_new_omega.append([])
        for j in range(len(new_omega_for_rect[i]) - 1):
            return_new_omega[i].append([new_omega_for_rect[i][j], new_omega_for_rect[i][j + 1]])
    tmp_new_size = 1
    for i in range(len(new_omega_for_rect)):
        tmp_new_size *= len(new_omega_for_rect[i])

    return return_new_omega, tmp_new_size


def intergate_list_of_rules_on_tmp_new_omega(z, old_omega, tmp_new_omega, list_of_rules,
                                             print_time_of_integration_for_each_rect):
    if print_time_of_integration_for_each_rect == True:
        print("     integrate z di for {} rules".format(list_of_rules))
    size_of_new_z_tensor = []
    for i in range(len(tmp_new_omega)):
        size_of_new_z_tensor.append(len(tmp_new_omega[i]))
    new_z = np.zeros(tuple(size_of_new_z_tensor))

    list_of_args = np.zeros(len(size_of_new_z_tensor), )

    start_time = 0
    if print_time_of_integration_for_each_rect == True:
        start_time = time.time()

    for i_1 in range(size_of_new_z_tensor[0]):
        for i_2 in range(size_of_new_z_tensor[1]):
            for i_3 in range(size_of_new_z_tensor[2]):
                list_of_args[0] = tmp_new_omega[0][i_1][0] + (
                        tmp_new_omega[0][i_1][1] - tmp_new_omega[0][i_1][0]) / 2
                list_of_args[1] = tmp_new_omega[1][i_2][0] + (
                        tmp_new_omega[1][i_2][1] - tmp_new_omega[1][i_2][0]) / 2
                list_of_args[2] = tmp_new_omega[2][i_3][0] + (
                        tmp_new_omega[2][i_3][1] - tmp_new_omega[2][i_3][0]) / 2
                sum = 0.0
                for p_index in list_of_rules:
                    sum += compute_z(z, old_omega, list_of_args, p_index)
                new_z[i_1][i_2][i_3] = sum
    if print_time_of_integration_for_each_rect == True:
        print("     time of integration: {} sek".format(time.time() - start_time))
        # print("########################################################################")
    return new_z


def get_indicies(input, new_omega):
    indicies = []
    for i in range(len(input)):
        for j in range(len(new_omega[i])):
            if ((input[i] >= new_omega[i][j][0]) and (input[i] <= new_omega[i][j][1])):
                indicies.append(j)
                break
    return tuple(indicies)


def get_output(new_omega, output_indices):
    output_list = []
    for i in range(len(output_indices)):
        output_list.append(new_omega[i][output_indices[i]][0] + (
                new_omega[i][output_indices[i]][1] - new_omega[i][output_indices[i]][0]) / 2)
    return output_list


def get_indicies_in_old_omega(input, old_omega, p):
    indicies = []
    for i in range(len(input)):
        for j in range(len(old_omega[p][i])):
            if (input[i] >= old_omega[p][i][j][0]) and (input[i] <= old_omega[p][i][j][1]):
                indicies.append(j)
                break
    return tuple(indicies)


def get_output_in_old_omega(old_omega, output_indices, p):
    output_list = []
    for i in range(len(output_indices)):
        output_list.append(old_omega[p][i][output_indices[i]][0] + (
                old_omega[p][i][output_indices[i]][1] - old_omega[p][i][output_indices[i]][0]) / 2)
    return output_list


def eval_list_of_integrals(input, old_z, old_omega, list_for_integrate, new_omega_list, list_of_integrals_z_di):
    # input = np.asarray([Df, V])
    list_of_maxs = []
    list_of_argmaxs = []
    # list_of_maxs = [
    #  [max(tensor), argmax(tensor)],
    #  [max(tensor), argmax(tensor)]
    # и т.д.
    # ]
    commulative_index = -1
    for i in range(len(list_for_integrate)):
        in_this_rect = 1
        for j in range(len(input)):
            if input[j] < list_for_integrate[i][0][j][0] or input[j] > list_for_integrate[i][0][j][1]:
                in_this_rect = 0
                break
        if len(list_for_integrate[i][1]) != 1:
            commulative_index += 1
        if in_this_rect == 1:
            # tmp_tmp = list_for_integrate[i][1]
            if len(list_for_integrate[i][1]) == 1:
                p = list_for_integrate[i][1][0]
                # не проверялось для длины 1
                indicies_of_input_in_pth_tensor = get_indicies_in_old_omega(input, old_omega, p)
                tmp_max = np.max(old_z[p][indicies_of_input_in_pth_tensor])
                index = np.argmax(old_z[p][indicies_of_input_in_pth_tensor])
                shape = np.shape(old_z[p][indicies_of_input_in_pth_tensor])
                output_indices = indicies_of_input_in_pth_tensor + np.unravel_index(index, shape)
                output = get_output_in_old_omega(old_omega, output_indices, p)[len(input):]
                list_of_maxs.append(tmp_max)
                list_of_argmaxs.append(output)
            else:
                indicies_of_input_in_multiple_rules_tensor = get_indicies(input, new_omega_list[i])
                max_in_multiple_rules_tensor = np.max(
                    list_of_integrals_z_di[commulative_index][indicies_of_input_in_multiple_rules_tensor])
                index = np.argmax(list_of_integrals_z_di[commulative_index][indicies_of_input_in_multiple_rules_tensor])
                shape = np.shape(list_of_integrals_z_di[commulative_index][indicies_of_input_in_multiple_rules_tensor])
                output_indices = indicies_of_input_in_multiple_rules_tensor + np.unravel_index(index, shape)
                output = get_output(new_omega_list[i], output_indices)[len(input):]
                list_of_maxs.append(max_in_multiple_rules_tensor)
                list_of_argmaxs.append(output)

    final_output = list_of_argmaxs[np.argmax(list_of_maxs)]
    return final_output

def get_all_p_ot_z_with_fixed_x_y(input, old_z, old_omega, list_for_integrate, new_omega_list, list_of_integrals_z_di):
    all_p_ot_x_y_z = []
    commulative_index = -1
    for i in range(len(list_for_integrate)):
        in_this_rect = 1
        for j in range(len(input)):
            if input[j] < list_for_integrate[i][0][j][0] or input[j] > list_for_integrate[i][0][j][1]:
                in_this_rect = 0
                break
        if len(list_for_integrate[i][1]) != 1:
            commulative_index += 1
        if in_this_rect == 1:
            # tmp_tmp = list_for_integrate[i][1]
            if len(list_for_integrate[i][1]) == 1:
                p = list_for_integrate[i][1][0]
                # не проверялось для длины 1
                indicies_of_input_in_pth_tensor = get_indicies_in_old_omega(input, old_omega, p)
                all_p_ot_x_y_z.append([i,old_z[p][indicies_of_input_in_pth_tensor], old_omega[p][2]])
            else:
                indicies_of_input_in_multiple_rules_tensor = get_indicies(input, new_omega_list[i])
                all_p_ot_x_y_z.append([i,list_of_integrals_z_di[commulative_index][indicies_of_input_in_multiple_rules_tensor], new_omega_list[i][2]])
    return all_p_ot_x_y_z

def eval_list_of_integrals_take_max_only(input, old_z, old_omega, list_for_integrate, new_omega_list, list_of_integrals_z_di):
    # input = np.asarray([Df, V])
    list_of_maxs = []
    list_of_argmaxs = []
    # list_of_maxs = [
    #  [max(tensor), argmax(tensor)],
    #  [max(tensor), argmax(tensor)]
    # и т.д.
    # ]
    commulative_index = -1
    for i in range(len(list_for_integrate)):
        in_this_rect = 1
        for j in range(len(input)):
            if input[j] < list_for_integrate[i][0][j][0] or input[j] > list_for_integrate[i][0][j][1]:
                in_this_rect = 0
                break
        if len(list_for_integrate[i][1]) != 1:
            commulative_index += 1
        if in_this_rect == 1:
            # tmp_tmp = list_for_integrate[i][1]
            if len(list_for_integrate[i][1]) == 1:
                p = list_for_integrate[i][1][0]
                # не проверялось для длины 1
                indicies_of_input_in_pth_tensor = get_indicies_in_old_omega(input, old_omega, p)
                tmp_max = np.max(old_z[p][indicies_of_input_in_pth_tensor])
                index = np.argmax(old_z[p][indicies_of_input_in_pth_tensor])
                shape = np.shape(old_z[p][indicies_of_input_in_pth_tensor])
                output_indices = indicies_of_input_in_pth_tensor + np.unravel_index(index, shape)
                output = get_output_in_old_omega(old_omega, output_indices, p)[len(input):]
                list_of_maxs.append(tmp_max)
                list_of_argmaxs.append(output)
            else:
                indicies_of_input_in_multiple_rules_tensor = get_indicies(input, new_omega_list[i])
                max_in_multiple_rules_tensor = np.max(
                    list_of_integrals_z_di[commulative_index][indicies_of_input_in_multiple_rules_tensor])
                index = np.argmax(list_of_integrals_z_di[commulative_index][indicies_of_input_in_multiple_rules_tensor])
                shape = np.shape(list_of_integrals_z_di[commulative_index][indicies_of_input_in_multiple_rules_tensor])
                output_indices = indicies_of_input_in_multiple_rules_tensor + np.unravel_index(index, shape)
                output = get_output(new_omega_list[i], output_indices)[len(input):]
                list_of_maxs.append(max_in_multiple_rules_tensor)
                list_of_argmaxs.append(output)

    final_output = np.max(list_of_maxs)
    return final_output

def plot_cube(axes, x_1, x_2, y_1, y_2, z_1, z_2):
    cube_definition = [
        (x_1, y_1, z_1), (x_1, y_2, z_1), (x_2, y_1, z_1), (x_1, y_1, z_2)
    ]
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    # faces = Poly3DCollection(edges, linewidths=1, edgecolors='k',alpha=1)
    faces = Poly3DCollection(edges)
    faces.set_facecolor((0, 0, 1, 0.1))

    axes.add_collection3d(faces)

def plot_cube_for_rules(axes, x_1, x_2, y_1, y_2, z_1, z_2,color):
    cube_definition = [
        (x_1, y_1, z_1), (x_1, y_2, z_1), (x_2, y_1, z_1), (x_1, y_1, z_2)
    ]
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors=color,alpha=0,facecolors=None)
    # faces = Poly3DCollection(edges)
    # faces.set_facecolor()

    axes.add_collection3d(faces)


def plot_cube_with_color(axes, x_1, x_2, y_1, y_2, z_1, z_2, color,alpha):
    cube_definition = [
        (x_1, y_1, z_1), (x_1, y_2, z_1), (x_2, y_1, z_1), (x_1, y_1, z_2)
    ]
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=0.1, edgecolors='k', alpha=alpha)
    # faces = Poly3DCollection(edges)
    faces.set_facecolor(color)

    axes.add_collection3d(faces)


def plot_rect(ax, x_1, x_2, y_1, y_2, color):
    h_1 = x_2 - x_1
    h_2 = y_2 - y_1
    rect = matplotlib.patches.Rectangle((x_1, y_1), h_1, h_2, color=color)
    ax.add_patch(rect)
def plot_rect_for_rules(ax, x_1, x_2, y_1, y_2, color):
    h_1 = x_2 - x_1
    h_2 = y_2 - y_1
    rect = matplotlib.patches.Rectangle((x_1, y_1), h_1, h_2, fc ='none',ec=color)
    ax.add_patch(rect)

def plot_x_2_x_3_use_eval_list(name_of_gif_and_png, input, old_z, old_omega, list_for_integrate, new_omega_list,
                               list_of_integrals_z_di,
                               index):
    plt.rcParams["figure.figsize"] = [14, 7]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    x = []
    y = []
    z = []
    commulative_index = -1

    rects_info = []

    for i in range(len(list_for_integrate)):
        in_this_rect = 1
        for j in range(len(input)):
            if input[j] < list_for_integrate[i][0][j][0] or input[j] > list_for_integrate[i][0][j][1]:
                in_this_rect = 0
                break
        if len(list_for_integrate[i][1]) != 1:
            commulative_index += 1
        if in_this_rect == 1:
            if len(list_for_integrate[i][1]) == 1:
                p = list_for_integrate[i][1][0]
                # не проверялось для длины 1
                indicies_of_input_in_pth_tensor = get_indicies_in_old_omega(input, old_omega, p)
                from_j_list = []
                to_j_list = []
                # как в to_j_list могла попасть 1
                for axis in range(len(list_for_integrate[i][0])):
                    j_min = 0
                    j_max = 0
                    if len(old_omega[p][axis]) == 1:
                        j_min = 0
                        j_max = 0
                    else:
                        for i_of_segment in range(len(old_omega[p][axis])):
                            # если риска попала в один из отрезков разбиения - запомним это
                            if old_omega[p][axis][i_of_segment][0] <= list_for_integrate[i][0][axis][0] <= \
                                    old_omega[p][axis][i_of_segment][1]:
                                j_min = i_of_segment
                                for i_of_following_segment in range(i_of_segment, len(old_omega[p][axis])):
                                    if old_omega[p][axis][i_of_following_segment][0] <= list_for_integrate[i][0][axis][
                                        1] <= \
                                            old_omega[p][axis][i_of_following_segment][1]:
                                        j_max = i_of_following_segment
                                        break
                                break
                    from_j_list.append(j_min)
                    to_j_list.append(j_max)
                for x_2_index in range(from_j_list[1], to_j_list[1] + 1):
                    for x_3_index in range(from_j_list[2], to_j_list[2] + 1):
                        # x.append(old_omega[p][1][x_2_index][0] + (
                        #         old_omega[p][1][x_2_index][1] - old_omega[p][1][x_2_index][0]) / 2)
                        # y.append(old_omega[p][2][x_3_index][0] + (
                        #         old_omega[p][2][x_3_index][1] - old_omega[p][2][x_3_index][0]) / 2)
                        z.append(old_z[p][indicies_of_input_in_pth_tensor][x_2_index][x_3_index])
                        # plot_cube(ax,old_omega[p][1][x_2_index][0],old_omega[p][1][x_2_index][1],old_omega[p][2][x_3_index][0],old_omega[p][2][x_3_index][1],0,old_z[p][indicies_of_input_in_pth_tensor][x_2_index][x_3_index])
                        rects_info.append([old_omega[p][1][x_2_index][0], old_omega[p][1][x_2_index][1],
                                           old_omega[p][2][x_3_index][0], old_omega[p][2][x_3_index][1],
                                           old_z[p][indicies_of_input_in_pth_tensor][x_2_index][x_3_index]])
            else:
                indicies_of_input_in_multiple_rules_tensor = get_indicies(input, new_omega_list[i])
                z_x_2_x_3 = list_of_integrals_z_di[commulative_index][indicies_of_input_in_multiple_rules_tensor]
                x_1_x_2_x_3 = new_omega_list[i]
                for x_2_index in range(len(x_1_x_2_x_3[1])):
                    for x_3_index in range(len(x_1_x_2_x_3[2])):
                        # x.append(x_1_x_2_x_3[1][x_2_index][0] + (
                        #         x_1_x_2_x_3[1][x_2_index][1] - x_1_x_2_x_3[1][x_2_index][0]) / 2)
                        # y.append(x_1_x_2_x_3[2][x_3_index][0] + (
                        #         x_1_x_2_x_3[2][x_3_index][1] - x_1_x_2_x_3[2][x_3_index][0]) / 2)
                        z.append(z_x_2_x_3[x_2_index][x_3_index])
                        # plot_rect(ax, x_1_x_2_x_3[1][x_2_index][0],x_1_x_2_x_3[1][x_2_index][1],x_1_x_2_x_3[2][x_3_index][0],x_1_x_2_x_3[2][x_3_index][1],m.to_rgba(z_x_2_x_3[x_2_index][x_3_index]))
                        rects_info.append(
                            [x_1_x_2_x_3[1][x_2_index][0], x_1_x_2_x_3[1][x_2_index][1], x_1_x_2_x_3[2][x_3_index][0],
                             x_1_x_2_x_3[2][x_3_index][1], z_x_2_x_3[x_2_index][x_3_index]])
                        # plot_cube(ax, x_1_x_2_x_3[1][x_2_index][0],x_1_x_2_x_3[1][x_2_index][1],x_1_x_2_x_3[2][x_3_index][0],x_1_x_2_x_3[2][x_3_index][1],0,z_x_2_x_3[x_2_index][x_3_index])

    norm = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
    cmap = cm.plasma
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    for i in range(len(rects_info)):
        plot_cube_with_color(ax, rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],rects_info[i][4],rects_info[i][4],
                  m.to_rgba(rects_info[i][4]))

    # plt.colorbar(m)
    # pnt3d = ax.scatter(x, y, z,c=z, alpha=0.9)
    plt.xlabel("скорость км/ч")
    plt.ylabel("сила нажатия на педаль")
    # cbar = plt.colorbar(pnt3d)
    # cbar.set_label("Values (units)")
    tmp_1 = str(input[0])
    if (len(tmp_1) > 4):
        tmp_1 = tmp_1[:4]
    plt.title("расстояние до стоп линии : {} см".format(tmp_1))
    ax.set_xlim([0.0, 94.0])
    ax.set_ylim([-1.0, 1.0])
    # ax.set_zlim([0.0, max(z)])
    path_to_save = "D:\\saved_fig\\" + name_of_gif_and_png + "_" + str(index) + ".png"
    plt.savefig(path_to_save)
    # газ и тормоз нажимаются только раздельно поэтому рисует только вдоль осей XZ и YZ
    plt.close(fig=fig)
    # plt.show()


def show_previous_best(filename):
    pprint.pprint(torch.load(filename))


def clear_dir_for_chat_with_sim(path_to_chating):
    for root, dirs, files in os.walk(path_to_chating):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def compute_a_h_f_coeff_list(rules, omega):
    f = []
    for rule in rules:
        f.append([])
        for distribution in rules[rule]:
            vec_of_f_k = torch.zeros(size=(rules[rule][distribution]["N"],), requires_grad=False)
            # проходимся по отрезкам и считаем значение функции на нем
            # значение счиатем на середине отрезка
            for i in range(rules[rule][distribution]["N"]):
                vec_of_f_k[i] = compute_f(rules[rule][distribution]["func"], omega[rule - 1][distribution - 1][i])
            f[rule - 1].append(vec_of_f_k)

    h = []
    for rule in rules:
        h.append([])
        for distribution in rules[rule]:
            razb = rules[rule][distribution]["razb"]
            h_vec = np.zeros(rules[rule][distribution]["N"], )
            for i in range(rules[rule][distribution]["N"]):
                h_vec[i] = razb[i + 1] - razb[i]
            h[rule - 1].append(torch.tensor(h_vec, requires_grad=False))

    a = []
    for rule in rules:
        size_of_pth_tensor = []
        for distribution in rules[rule]:
            size_of_pth_tensor.append(rules[rule][distribution]["N"])
        a_p_np = np.zeros(shape=tuple(size_of_pth_tensor))

        for k_1 in range(size_of_pth_tensor[0]):
            for k_2 in range(size_of_pth_tensor[1]):
                for k_3 in range(size_of_pth_tensor[2]):
                    a_p_np[k_1][k_2][k_3] = h[rule - 1][0][k_1] * h[rule - 1][1][k_2] * h[rule - 1][2][k_3]
        a.append(torch.from_numpy(a_p_np))

    coeff_list = []
    for rule in rules:
        coeff_list.append([])
        for distribution in rules[rule]:
            coeff_list[rule - 1].append((rules[rule][distribution]["b"] - rules[rule][distribution]["a"]) ** 2)

    return a, h, f, coeff_list


def make_list_for_integrate(omega_for_rules, tensor_with_info):
    list_for_integrate = []

    #   пройдемся по всем элементам tensor_with_information_about_intersection и проинтегрируем там где не пустой элемент
    for i_1 in range(len(omega_for_rules[0])):
        for i_2 in range(len(omega_for_rules[1])):
            for i_3 in range(len(omega_for_rules[2])):
                if len(tensor_with_info[i_1][i_2][i_3]) > 0:
                    # в области определения соответствующей прямоугольнику с индексами i_1 i_2 i_3 i_4
                    # пересеклись опредлененные правила. их список мы знаем. это может быть и одно правило, тогда
                    # длина списка = 1, но если длина списка = 1 то не нужно ни копировать ,ни интегрировать
                    # если длина списка > 1 то нужно посчитать интеграл
                    # зная индексы мы можем сказать левые и правые границы по каждой из осей этого прямоугольника
                    list_for_integrate.append(
                        [
                            [omega_for_rules[0][i_1], omega_for_rules[1][i_2], omega_for_rules[2][i_3]],

                            tensor_with_info[i_1][i_2][i_3]
                        ]
                    )
    return list_for_integrate


def make_new_omega_list(np_omega, list_for_integrate, print_num_of_rect):
    new_omega_list = []
    num_of_rect_in_intersection_list = []
    for i in range(len(list_for_integrate)):
        tmp_new_omega, num_of_rect_in_intersection = make_new_omega_for_rect(old_omega=np_omega,
                                                                             rect=list_for_integrate[i][0],
                                                                             list_of_rules=list_for_integrate[i][1])
        new_omega_list.append(tmp_new_omega)
        num_of_rect_in_intersection_list.append(num_of_rect_in_intersection)
    if print_num_of_rect == True:
        print("     num_of_rect_in_intersection {}".format(sum(num_of_rect_in_intersection_list)))
    return new_omega_list


def make_omega_for_init_z(rules, num_of_rules, dimension):
    a_i_j = np.zeros(shape=(num_of_rules, dimension))
    b_i_j = np.zeros(shape=(num_of_rules, dimension))

    x_i_vec = []
    y_i_vec = []
    z_i_vec = []
    for rule in rules:
        size_of_pth_tensor = []
        for distribution in rules[rule]:
            a_i_j[rule - 1][distribution - 1] = rules[rule][distribution]["a"]
            b_i_j[rule - 1][distribution - 1] = rules[rule][distribution]["b"]
            size_of_pth_tensor.append(rules[rule][distribution]["N"])
        x_i_vec.append(np.zeros(size_of_pth_tensor[0], ))
        y_i_vec.append(np.zeros(size_of_pth_tensor[1], ))
        z_i_vec.append(np.zeros(size_of_pth_tensor[2], ))
        for i in range(size_of_pth_tensor[0]):
            x_i_vec[rule - 1][i] = (rules[rule][1]["razb"][i + 1] + rules[rule][1]["razb"][i]) / 2
        for i in range(size_of_pth_tensor[1]):
            y_i_vec[rule - 1][i] = (rules[rule][2]["razb"][i + 1] + rules[rule][2]["razb"][i]) / 2
        for i in range(size_of_pth_tensor[2]):
            z_i_vec[rule - 1][i] = (rules[rule][3]["razb"][i + 1] + rules[rule][3]["razb"][i]) / 2
    return a_i_j, b_i_j, x_i_vec, y_i_vec, z_i_vec


def make_init_z(num_of_rules, distrib_of_rules, dimension, a_i_j, b_i_j,random_seed, x_i_vec, y_i_vec, z_i_vec, razb_tensor_a,list_of_rect):
    init_z = []  # не прологарифмированные значения высот:
    N = random_seed["N"]
    c_i_j_k=random_seed["c_i_j_k"]
    mu_x=random_seed["mu_x"]
    mu_y=random_seed["mu_y"]
    mu_z=random_seed["mu_z"]
    sigma_x=random_seed["sigma_x"]
    sigma_y=random_seed["sigma_y"]
    sigma_z=random_seed["sigma_z"]
    for i in range(num_of_rules):
        x_from_torch = torch.from_numpy(x_i_vec[i]).view(len(x_i_vec[i]), 1)
        y_from_torch = torch.from_numpy(y_i_vec[i]).view(len(y_i_vec[i]), 1)
        z_from_torch = torch.from_numpy(z_i_vec[i]).view(len(z_i_vec[i]), 1)
        not_normed_z = compute_seq_vec(x_from_torch, y_from_torch, z_from_torch, c_i_j_k, N, a_i_j[i][0], a_i_j[i][1],
                                       a_i_j[i][2],
                                       b_i_j[i][0], b_i_j[i][1],
                                       b_i_j[i][2]) * compute_gauss(x_from_torch, y_from_torch, z_from_torch,
                                                                    mu_x[i],
                                                                    sigma_x[i],
                                                                    mu_y[i],
                                                                    sigma_y[i],
                                                                    mu_z[i],
                                                                    sigma_z[i])
        # not_normed_z = compute_gauss(x_from_torch, y_from_torch, z_from_torch, mu_x,
        #                                                             sigma_x, mu_y,
        #                                                             sigma_y,
        #                                                             mu_z, sigma_z)
        # необходимо отнормировать эту функцию, чтобы она стала распределением
        norm = torch.sum(not_normed_z * razb_tensor_a[i])
        normed_z = not_normed_z / norm
        init_z.append(normed_z * distrib_of_rules[i])
    # init_z = []  # не прологарифмированные значения высот:
    # N = 1
    # c_i_j_k = np.random.rand(2 * N + 1, 2 * N + 1, 2 * N + 1)
    # # найдем границы по каждой оси
    # axis_left_sides = []
    # axis_right_sides = []
    # for axis in range(len(list_of_rect[0])):
    #     axis_left_sides.append([])
    #     axis_right_sides.append([])
    #     for rect_index in range(len(list_of_rect)):
    #         axis_left_sides[axis].append(list_of_rect[rect_index][axis][0])
    #         axis_right_sides[axis].append(list_of_rect[rect_index][axis][1])
    # axis_left_sides = np.asarray(axis_left_sides)
    # axis_right_sides = np.asarray(axis_right_sides)
    # mins_of_x_i = np.zeros(len(axis_left_sides), )
    # maxs_of_x_i = np.zeros(len(axis_right_sides), )
    # for axis in range(len(axis_left_sides)):
    #     mins_of_x_i[axis] = np.min(axis_left_sides[axis])
    #     maxs_of_x_i[axis] = np.max(axis_right_sides[axis])
    # x_a = mins_of_x_i[0]
    # x_b = maxs_of_x_i[0]
    # y_a= mins_of_x_i[1]
    # y_b = maxs_of_x_i[1]
    # z_a = mins_of_x_i[2]
    # z_b = maxs_of_x_i[2]
    # 
    # mu_x = np.random.uniform(low=(x_a + x_b) * 0.25, high=(x_a + x_b) * 0.75)
    # mu_y = np.random.uniform(low=(y_a + y_b) * 0.25, high=(y_a + y_b) * 0.75)
    # mu_z = np.random.uniform(low=(z_a + z_b) * 0.25, high=(z_a + z_b) * 0.75)
    # sigma_x = (x_b - x_a) / 2
    # sigma_y = (y_b - y_a) / 2
    # sigma_z = (z_b - z_a) / 2
    # 
    # for i in range(num_of_rules):
    #     x_from_torch = torch.from_numpy(x_i_vec[i]).view(len(x_i_vec[i]), 1)
    #     y_from_torch = torch.from_numpy(y_i_vec[i]).view(len(y_i_vec[i]), 1)
    #     z_from_torch = torch.from_numpy(z_i_vec[i]).view(len(z_i_vec[i]), 1)
    #     not_normed_z = compute_seq_vec(x_from_torch, y_from_torch, z_from_torch, c_i_j_k, N, x_a, y_a,
    #                                    z_a,
    #                                    x_b, y_b,
    #                                    z_b) * compute_gauss(x_from_torch, y_from_torch, z_from_torch, mu_x,
    #                                                                 sigma_x, mu_y,
    #                                                                 sigma_y,
    #                                                                 mu_z, sigma_z)
    #     # необходимо отнормировать эту функцию, чтобы она стала распределением
    #     norm = torch.sum(not_normed_z * razb_tensor_a[i])
    #     normed_z = not_normed_z / norm
    #     init_z.append(normed_z * distrib_of_rules[i])
    return init_z


def make_RO_tensor(x_i_vec, y_i_vec, z_i_vec, num_of_rules, dimension):
    print("start make ro")
    start_time = time.time()
    X_i_j_k_p = []
    for i in range(num_of_rules):
        x_size = len(x_i_vec[i])
        y_size = len(y_i_vec[i])
        z_size = len(z_i_vec[i])
        X_j_k_p = torch.zeros(size=(x_size, y_size, z_size, dimension, 1))
        for j in range(x_size):
            for k in range(y_size):
                for p in range(z_size):
                    X_j_k_p[j][k][p][0] = x_i_vec[i][j]
                    X_j_k_p[j][k][p][1] = y_i_vec[i][k]
                    X_j_k_p[j][k][p][2] = z_i_vec[i][p]
        X_i_j_k_p.append(X_j_k_p)
    stop_time = time.time()
    print(stop_time - start_time)
    print("end make ro")
    print(1)


def plot_multiple_z_after_training_and_init_z(num_of_fig, mode_of_plot, filepath_to_fig_and_gifs, z_init,
                                              solutions_from_same_init_z, list_of_rect, old_omega,
                                              distrib_of_init_rules, train_dicts):
    dfs = np.linspace(0, 6500, num_of_fig)

    if mode_of_plot == "rect":
        print("making figures")
        os.makedirs(filepath_to_fig_and_gifs + "\\figures")

        for figure_index in range(num_of_fig):
            print(figure_index)
            plt.rcParams["figure.figsize"] = [28, 14]
            plt.rcParams["figure.autolayout"] = True
            fig, axs = plt.subplots(len(solutions_from_same_init_z) + 1)
            print("next cadr")
            # x = []
            # y = []
            z_init_highs = []
            sizes_of_marker = []
            z_solutions_highs = []
            for solution_index in range(len(solutions_from_same_init_z)):
                z_solutions_highs.append([])
            rects_info = []

            input = dfs[figure_index]
            # узнаем, в какие правила попадает входное значение
            in_this_rules = np.zeros(len(list_of_rect), )
            for i in range(len(list_of_rect)):
                in_this_rect = 1
                if input < list_of_rect[i][0][0] or input > list_of_rect[i][0][1]:
                    in_this_rect = 0
                in_this_rules[i] = in_this_rect
            print(in_this_rules)
            for i in range(len(list_of_rect)):
                if in_this_rules[i] == 1:
                    # в тех правилах, в которые попало значения dfs[figure_index] найдем индексы соответсвующие этому значению
                    # и построим сразу и изначальное распределение и полученные градиентом решения
                    indicies_of_input_in_pth_tensor = get_indicies_in_old_omega([input], old_omega, i)

                    for x_2_index in range(len(old_omega[i][1])):
                        for x_3_index in range(len(old_omega[i][2])):
                            x_1 = old_omega[i][1][x_2_index][0]
                            x_2 = old_omega[i][1][x_2_index][1]
                            y_1 = old_omega[i][2][x_3_index][0]
                            y_2 = old_omega[i][2][x_3_index][1]
                            # mid_point_x = (x_1+x_2)/2
                            # mid_point_y = (y_1+y_2)/ 2
                            # x.append(mid_point_x)
                            # y.append(mid_point_y)
                            rects_info.append([x_1, x_2, y_1, y_2])
                            z_init_highs.append(z_init[i][indicies_of_input_in_pth_tensor][x_2_index][x_3_index] *
                                                distrib_of_init_rules[i])
                            for solution_index in range(len(solutions_from_same_init_z)):
                                z_solutions_highs[solution_index].append(
                                    solutions_from_same_init_z[solution_index][i][indicies_of_input_in_pth_tensor][
                                        x_2_index][x_3_index])

            norm_1 = matplotlib.colors.Normalize(vmin=min(z_init_highs), vmax=max(z_init_highs))
            cmap_1 = cm.jet
            m_1 = cm.ScalarMappable(norm=norm_1, cmap=cmap_1)

            m_i_list = []
            for solution_index in range(len(solutions_from_same_init_z)):
                norm_i = matplotlib.colors.Normalize(vmin=min(z_solutions_highs[solution_index]),
                                                     vmax=max(z_solutions_highs[solution_index]))
                cmap_i = cm.jet
                m_i = cm.ScalarMappable(norm=norm_i, cmap=cmap_i)
                m_i_list.append(m_i)

            for i in range(len(rects_info)):
                plot_rect(axs[0], rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],
                          m_1.to_rgba(z_init_highs[i]))
                for solution_index in range(len(solutions_from_same_init_z)):
                    plot_rect(axs[solution_index + 1], rects_info[i][0], rects_info[i][1], rects_info[i][2],
                              rects_info[i][3],
                              m_i_list[solution_index].to_rgba(z_solutions_highs[solution_index][i]))

            # compute limits
            min_x = 0
            max_x = 0
            min_y = 0
            max_y = 0

            for i in range(len(in_this_rules)):
                if in_this_rules[i] == True:
                    min_x = list_of_rect[i][1][0]
                    max_x = list_of_rect[i][1][1]
                    min_y = list_of_rect[i][2][0]
                    max_y = list_of_rect[i][2][1]
                    break

            for i in range(len(in_this_rules)):
                if in_this_rules[i] == True:
                    min_x = min(list_of_rect[i][1][0], min_x)
                    max_x = max(list_of_rect[i][1][1], max_x)
                    min_y = min(list_of_rect[i][2][0], min_y)
                    max_y = max(list_of_rect[i][2][1], max_y)

            # scatter_line = axs[0].scatter(x,y,c=z_init_highs,alpha=0.9)

            # cbar = plt.colorbar(mappable=m_1, ax=axs[0], format='%.0e')
            # cbar.set_label("Values (units)")
            # axs[0].set_xlim([0.0, 94.0])
            # axs[0].set_ylim([-1.0, 1.0])
            axs[0].set_xlim([min_x, max_x])
            axs[0].set_ylim([min_y, max_y])
            # axs[0].set_xlabel("скорость км/ч")
            # axs[0].set_ylabel("сила нажатия на педаль")
            # axs[0].set_title("init_z")

            for solution_index in range(len(solutions_from_same_init_z)):
                # tmp_cbar = plt.colorbar(mappable=m_i_list[solution_index], ax=axs[solution_index + 1], format='%.0e')
                # tmp_cbar.set_label("Values (units)")
                axs[solution_index + 1].set_xlim([min_x, max_x])
                axs[solution_index + 1].set_ylim([min_y, max_y])
                # axs[solution_index+1].set_xlabel("скорость км/ч")
                # axs[solution_index+1].set_ylabel("сила нажатия на педаль")
                axs[solution_index + 1].set_title("{}".format(train_dicts[solution_index]["mode_of_loss"]))

            fig.suptitle(
                "расстояние до стоп линии : {:10.0f} см. задействованные правила {}".format(input, in_this_rules))
            path_to_save = filepath_to_fig_and_gifs + "\\figures\\figure_" + str(figure_index) + ".png"
            plt.savefig(path_to_save)
            plt.close(fig=fig)
            # plt.show()
    print("making figures done")
    print("making gif")
    os.makedirs(filepath_to_fig_and_gifs + "\\gif")
    with imageio.get_writer(filepath_to_fig_and_gifs + "\\gif\\gif.gif", mode='I') as writer:
        for i in range(num_of_fig):
            print(i)
            image = imageio.imread(filepath_to_fig_and_gifs + "\\figures\\figure_" + str(i) + ".png")
            writer.append_data(image)
    print("making gif done")


def make_distrib_of_rules(num_of_rules):
    not_normed_distrib = np.random.rand(num_of_rules, )
    norm = np.sum(not_normed_distrib)
    return not_normed_distrib / norm

def get_projection_to_x_of_x_y_grid(list_for_integrate,new_omega_list):
    unique_x_areas = {}
    for i in range(len(list_for_integrate)):
        rect = list_for_integrate[i][0]
        x_area_of_rect = rect[0]
        # проверим, не проверяли ли мы уже эту область в словаре unique_x_y_areass
        already_in_dict = 0
        for key in unique_x_areas.keys():
            x_in_rect = unique_x_areas[key]
            if x_in_rect[0] == x_area_of_rect[0] and x_in_rect[1] == x_area_of_rect[1]:
                already_in_dict = 1
                break
        if already_in_dict == 1:
            continue
        else:
            unique_x_areas.update({i: rect[0]})

    areas_of_x = []
    for key in unique_x_areas.keys():
        x_range = unique_x_areas[key]
        areas_of_x.append(x_range)
    output = []
    for i in range(len(areas_of_x)):
        x_of_reference_area = areas_of_x[i]
        # найдем индексы всех областей которые попадат в данный прямоугольник
        areas_that_contain_x_area = []
        for j in range(len(list_for_integrate)):
            rect = list_for_integrate[j][0]
            x_of_tmp_area = rect[0]
            if x_of_reference_area[0] == x_of_tmp_area[0] and x_of_reference_area[1] == x_of_tmp_area[1]:
                areas_that_contain_x_area.append(j)

        grids_of_x = []
        for index_of_area in areas_that_contain_x_area:
            grids_of_x.append(new_omega_list[index_of_area][0])
        grid_of_x = []
        for j in range(len(grids_of_x)):
            for k in range(len(grids_of_x[j])):
                grid_of_x.append(grids_of_x[j][k][0])
                grid_of_x.append(grids_of_x[j][k][1])
        superposition_of_x = np.unique(grid_of_x)

        superposition_of_grids_in_reference_area = []
        for j in range(len(superposition_of_x) - 1):
            superposition_of_grids_in_reference_area.append([superposition_of_x[j], superposition_of_x[j + 1]])
        output.append([x_of_reference_area, areas_that_contain_x_area, superposition_of_grids_in_reference_area])
    return output

def compute_F(projection_to_x_y_info,list_of_z_and_p_with_fixed_x_y):
    F = []
    Rects = []
    Grids = []
    for i in range(len(projection_to_x_y_info)):
        info = projection_to_x_y_info[i]
        superposition_of_grids_in_reference_area = info[2]
        Rects.append(info[0])
        F.append([])
        Grids.append(superposition_of_grids_in_reference_area)
        for j in range(len(superposition_of_grids_in_reference_area[0])):
            F[i].append([])
            for k in range(len(superposition_of_grids_in_reference_area[1])):
                all_z = list_of_z_and_p_with_fixed_x_y[i][0][j][k]
                all_p = list_of_z_and_p_with_fixed_x_y[i][1][j][k]
                tmp_max = np.max(all_p)
                tmp_argmax = np.where(all_p == tmp_max)
                z_otrezok_of_answer = all_z[tmp_argmax]
                F[i][j].append(z_otrezok_of_answer)
    return F,Rects,Grids

def get_information_about_neighbors(Rects):
    Neighbors = {}
    for i in range(len(Rects)):
        neighbours_for_current_rect = []
        current_rect = Rects[i]
        x_of_current_area = current_rect[0]
        a_x = x_of_current_area[0]
        b_x = x_of_current_area[1]
        y_of_current_area = current_rect[1]
        a_y = y_of_current_area[0]
        b_y = y_of_current_area[1]
        current_vertexes = [[a_x,a_y],[a_x,b_y],[b_x,b_y],[b_x,a_y]]
        # найдем соседей заданного прямоугольника
        for j in range(len(Rects)):
            if i != j:
                tmp_rect = Rects[j]
                x_of_tmp_area= tmp_rect[0]
                a_x_tmp = x_of_tmp_area[0]
                b_x_tmp = x_of_tmp_area[1]
                y_of_tmp_area= tmp_rect[1]
                a_y_tmp = y_of_tmp_area[0]
                b_y_tmp = y_of_tmp_area[1]
                tmp_vertexes = [[a_x_tmp, a_y_tmp], [a_x_tmp, b_y_tmp], [b_x_tmp, b_y_tmp], [b_x_tmp, a_y_tmp]]
                # если хотя бы одна вершина совпала, то это соседи
                is_neighbor = False
                for k in range(len(current_vertexes)):
                    if is_neighbor==True:
                        break
                    current_vertex = current_vertexes[k]
                    for p in range(len(tmp_vertexes)):
                        if is_neighbor==True:
                            break
                        tmp_vertex = tmp_vertexes[p]
                        if (current_vertex[0] == tmp_vertex[0])and(current_vertex[1] == tmp_vertex[1]):
                            neighbours_for_current_rect.append(j)
                            is_neighbor =True
                            break

        Neighbors.update({i:neighbours_for_current_rect})
    return Neighbors

def get_global_rect_X_Y(X_Y_Z_rects):
    x=[]
    y=[]
    for i in range(len(X_Y_Z_rects)):
        x.append(X_Y_Z_rects[i][0][0])
        x.append(X_Y_Z_rects[i][0][1])
        y.append(X_Y_Z_rects[i][1][0])
        y.append(X_Y_Z_rects[i][1][1])
    rect_of_x_y =[[np.min(x), np.max(x)],[np.min(y),np.max(y)]]
    return rect_of_x_y

def where_window(x_area,y_area,Rects):
    list_of_areas = []
    # вершины по часовой стрелке
    current_vertexes = [[x_area[0],y_area[0]],[x_area[0],y_area[1]],[x_area[1],y_area[1]],[x_area[1],y_area[0]]]
    for i in range(len(Rects)):
        tmp_area = Rects[i]
        tmp_x_area = tmp_area[0]
        tmp_y_area = tmp_area[1]
        # если хоть одна вершина попала в этот прямоугольник то запишем это
        for j in range(len(current_vertexes)):
            tmp_vertex = current_vertexes[j]
            x_of_vertex = tmp_vertex[0]
            y_of_vertex = tmp_vertex[1]
            if  (tmp_x_area[0]<= x_of_vertex)and(x_of_vertex<=tmp_x_area[1])and (tmp_y_area[0]<= y_of_vertex)and(y_of_vertex<=tmp_y_area[1]):
                list_of_areas.append(i)
                break
    return list_of_areas

def get_values_by_x_y_in_fixed_area(x_area,y_area,surf,Rects,Grids,index_of_area):
    # необходимо пройтись по всем прямоугольникам в этой области, которые попали в окно шириной d_x,d_y
    # и взять минимальное значение
    current_vertexes = [[x_area[0],y_area[0]],[x_area[0],y_area[1]],[x_area[1],y_area[1]],[x_area[1],y_area[1]]]
    Grid = Grids[index_of_area]
    Rect = Rects[index_of_area]
    piece_of_surf = surf[index_of_area]
    # примитивный подход- пройдемся по всем прямоугольникам и проверим не попал ли какой то конкретный в эту область
    values =[]
    rects_of_values=[]
    for i in range(len(Grid[0])):
        tmp_x_area = Grid[0][i]
        for j in range(len(Grid[1])):
            tmp_y_area = Grid[1][j]
            vertexes = [[tmp_x_area[0],tmp_y_area[0]],[tmp_x_area[0],tmp_y_area[1]],[tmp_x_area[1],tmp_y_area[1]],[tmp_x_area[1],tmp_y_area[0]]]
            # если хоть одна вершина попала в d_x,d_y окно, то нужно взять значение
            already_find = False
            for k in range(len(vertexes)):
                x_of_vertex = vertexes[k][0]
                y_of_vertex = vertexes[k][1]
                if (x_of_vertex>=x_area[0]) and (x_of_vertex<=x_area[1])and(y_of_vertex>=y_area[0])and(y_of_vertex<=y_area[1]):
                    values.append(piece_of_surf[i][j])
                    rects_of_values.append([tmp_x_area, tmp_y_area])
                    already_find = True
                    break
            if already_find==False:
                for k in range(len(current_vertexes)):
                    x_of_vertex = current_vertexes[k][0]
                    y_of_vertex = current_vertexes[k][1]
                    if (x_of_vertex>=tmp_x_area[0]) and (x_of_vertex<=tmp_x_area[1])and(y_of_vertex>=tmp_y_area[0])and(y_of_vertex<=tmp_y_area[1]):
                        values.append(piece_of_surf[i][j])
                        rects_of_values.append([tmp_x_area, tmp_y_area])
                        break
    if len(values)==0:
        print(1)
    return values,rects_of_values


def get_lip_by_x_y_area(x_area,y_area,surf,Rects,Grids):
    lips=[]
    # если хоть одна грань попала внутрь области то там нужно брать значения
    list_of_target_areas = where_window(x_area,y_area,Rects)
    all_values=[]
    all_rects =[]
    for i in range(len(list_of_target_areas)):
        tmp_area_index = list_of_target_areas[i]
        values,rects_of_values = get_values_by_x_y_in_fixed_area(x_area,y_area,surf,Rects,Grids,index_of_area=tmp_area_index)

        for j in range(len(values)):
            all_values.append(values[j])
            all_rects.append(rects_of_values[j])
    # проверим каждый прямоугольничек с каждым посчитаем аналог производной по центральным точкам пдвух прямоугольников
    for i in range(len(all_rects)):
        curren_rect = all_rects[i]
        curren_x_area = curren_rect[0]
        curren_y_area = curren_rect[1]
        current_mid_x = (curren_x_area[0]+curren_x_area[1])/2
        current_mid_y = (curren_y_area[0] + curren_y_area[1]) / 2
        curent_value = all_values[i]
        for j in range(len(all_rects)):
            if i!=j:
                tmp_rect = all_rects[j]
                tmp_x_area = tmp_rect[0]
                tmp_y_area = tmp_rect[1]
                tmp_mid_x = (tmp_x_area[0] + tmp_x_area[1]) / 2
                tmp_mid_y = (tmp_y_area[0] + tmp_y_area[1]) / 2
                tmp_value = all_values[j]
                ro = np.sqrt(np.square(current_mid_x-tmp_mid_x)+np.square(current_mid_y-tmp_mid_y))
                lips.append((np.abs(curent_value-tmp_value))/ro)

    if len(lips)==0:
        return 0
    else:
        return np.max(lips)

def compute_lip(surf,Rects,Grids,U,X_Y_Z_rects):
    d_x = U[0]
    d_y = U[1]
    x_rects=[]
    y_rects=[]
    lips = []

    # superposition_of_grids_in_reference_area = Grids[i]
    # z_otrezok_of_answer = F[i][j][k]
    X_Y_area = get_global_rect_X_Y(X_Y_Z_rects)
    # Neighbors = get_information_about_neighbors(Rects)
    x_pos = X_Y_area[0][0]
    all_steps_by_x = (X_Y_area[0][0]-X_Y_area[0][1])/d_x
    all_steps_by_y = (X_Y_area[1][0]-X_Y_area[1][1])/d_y
    num_of_step = all_steps_by_y*all_steps_by_x
    tmp_index= 0
    while x_pos < X_Y_area[0][1]:
        print("\r step {} of {} steps".format(tmp_index,num_of_step),end='')
        y_pos = X_Y_area[1][0]
        while y_pos < X_Y_area[1][1]:
            tmp_index+=1
            x_tmp = [x_pos,x_pos+d_x]
            y_tmp = [y_pos,y_pos+d_y]
            x_rects.append(x_tmp)
            y_rects.append(y_tmp)
            # находим минимальную константу Липшица
            lip_const = get_lip_by_x_y_area(x_tmp,y_tmp,surf,Rects,Grids)
            lips.append(lip_const)
            y_pos = y_pos+d_y
        x_pos = x_pos+d_x
    return x_rects,y_rects,lips

def loss_for_surf(F, Grids,ref_surf):

    pr = torch.zeros(1, requires_grad=False)
    ro_to_ref = torch.zeros(1, requires_grad=False)
    for i in range(len(Grids)):
        superposition_of_grids_in_reference_area = Grids[i]
        # сетка равномерная поэтому можно сэкономить время
        h_x = np.abs(superposition_of_grids_in_reference_area[0][0][0]-superposition_of_grids_in_reference_area[0][0][1])
        h_y = np.abs(superposition_of_grids_in_reference_area[1][0][0]-superposition_of_grids_in_reference_area[1][0][1])
        h_x_h_y = np.sqrt(np.square(h_x)+np.square(h_y))
        for j in range(1,len(superposition_of_grids_in_reference_area[0])-1,2):
            for k in range(1,len(superposition_of_grids_in_reference_area[1])-1,2):
                pr = pr + torch.square(
                     torch.abs(F[i][j][k - 1] - F[i][j][k])/h_y + \
                     torch.abs(F[i][j-1][k - 1] - F[i][j][k])/h_x_h_y + \
                     torch.abs(F[i][j-1][k] - F[i][j][k])/h_x + \
                     torch.abs(F[i][j-1][k+1] - F[i][j][k])/h_x_h_y + \
                     torch.abs(F[i][j][k+1] - F[i][j][k])/h_y + \
                     torch.abs(F[i][j+1][k+1] - F[i][j][k])/h_x_h_y + \
                     torch.abs(F[i][j+1][k] - F[i][j][k])/h_x + \
                     torch.abs(F[i][j+1][k - 1] - F[i][j][k])/h_x_h_y
                )
                ro_to_ref = ro_to_ref + torch.square(
                    torch.abs(ref_surf[i][j][k] - F[i][j][k])+\
                    torch.abs(ref_surf[i][j][k - 1] - F[i][j][k - 1]) + \
                    torch.abs(ref_surf[i][j - 1][k - 1] - F[i][j - 1][k - 1]) + \
                    torch.abs(ref_surf[i][j - 1][k] - F[i][j - 1][k]) + \
                    torch.abs(ref_surf[i][j - 1][k + 1] - F[i][j - 1][k + 1]) + \
                    torch.abs(ref_surf[i][j][k + 1] - F[i][j][k + 1]) + \
                    torch.abs(ref_surf[i][j + 1][k + 1] - F[i][j + 1][k + 1]) + \
                    torch.abs(ref_surf[i][j + 1][k] - F[i][j + 1][k]) + \
                    torch.abs(ref_surf[i][j + 1][k - 1] - F[i][j + 1][k - 1])
                    )

    return pr +ro_to_ref*0.5

def surf_optimization(F,Grids,Rects,BayesAreas):
    start_point_F = make_response_surface(F, Grids)
    # x_rects_for_lips, y_rects_for_lips, lips_values = compute_lip(start_point_F, Rects, Grids, U=[5, 0.25],
    #                                                               X_Y_Z_rects=list_of_rect)
    start_point_F_torch = []
    for i in range(len(start_point_F)):
        start_point_F_torch.append(torch.tensor(start_point_F[i],requires_grad=True))

    optimizer = torch.optim.Adam(start_point_F_torch, 0.01, [0.5, 0.7])
    num_of_steps = 400
    loss_vec = np.zeros(num_of_steps,)
    print("\n making oprimized surf")
    for i in range(num_of_steps):
        optimizer.zero_grad()
        loss_t = loss_for_surf(start_point_F_torch, Grids, start_point_F)
        loss_value = float(loss_t.detach().numpy())
        loss_vec[i]= loss_value
        loss_t.backward()
        optimizer.step()
        print("\r step {} of {} steps loss {}".format(i,num_of_steps,loss_value), end='')
    print("\n")
    fig_loss, axs_loss = plt.subplots(1, 1)
    loss_line, = axs_loss.plot(loss_vec)
    axs_loss.set_title("loss")
    axs_loss.set_yscale("linear")
    plt.show(block=True)
    usable_F = []
    for i in range(len(start_point_F)):
        usable_F.append(start_point_F_torch[i].detach().numpy())
    return usable_F

def make_response_surf_with_grad(F,Grids,Rects,BayesAreas):
    surf = surf_optimization(F,Grids,Rects,BayesAreas)
    return surf

def make_response_surface(F,Grids):
    surf = []
    for i in range(len(F)):
        surf.append([])
        superposition_of_grids_in_reference_area = Grids[i]
        for j in range(len(superposition_of_grids_in_reference_area[0])):
            surf[i].append([])
            for k in range(len(superposition_of_grids_in_reference_area[1])):
                z_otrezok_of_answer = F[i][j][k]
                surf[i][j].append( (z_otrezok_of_answer[0][0] + z_otrezok_of_answer[0][1]) / 2)
    return surf


def plot_response_surface(
        mode_of_plot,
        filepath_to_save_response_surface,F, Rects, Grids,list_for_integrate,block_canvas,
        distanses_to_line, v_vec,plot_rules,problems_areas_info,bayes_areas,quantile_for_bayes_areas,
        plot_lip,x_rects_for_lips,y_rects_for_lips,lips_values):
    print("\nstart plot response surface")
    # list_for_integrate[i][0]-края
    # list_for_integrate[i][1]-какое там распределение
    # new_omega_list[i][0-2][0-(N-1)] - разбиение по контретному направлению
    if plot_lip==True:
        plt.rcParams["figure.figsize"] = [14, 7]
        plt.rcParams["figure.autolayout"] = True
        fig_lip = plt.figure()
        axs_lip = fig_lip.add_subplot(111)

        norm = matplotlib.colors.Normalize(vmin=min(lips_values), vmax=max(lips_values))
        m = cm.ScalarMappable(norm=norm, cmap=cm.jet)

        tmp_len = len(x_rects_for_lips)
        for i in range(len(x_rects_for_lips)):
            print("\r interation {} of {}".format(i, tmp_len), end='')
            plot_rect(axs_lip, x_rects_for_lips[i][0], x_rects_for_lips[i][1], y_rects_for_lips[i][0], y_rects_for_lips[i][1],
                      m.to_rgba(lips_values[i]))

        plt.colorbar(m, ax=axs_lip)
        axs_lip.set_xlim([0, 361.1])
        axs_lip.set_ylim([0, 22.2])
        axs_lip.set_xlabel(r"$расстояние \:до \:стоп-линии  \: x_{1}, \: м$")
        axs_lip.set_ylabel(r'$скорость \: x_{2}, \: м \cdot c^{-1}$')
        axs_lip.set_title(r'"локальные" константы Липшица' '\n' r' максимальная "локальная" константа Липшица 'r'$ \mathbf{lip}_{\mathbf{X}}(F)=\:$'+str(max(lips_values))[:4])
        # +str(max(lips_values))
        plt.show(block=False)
        print("plot lip done")

    x = []
    y = []
    rects_info = []
    highs = []
    z_for_cubes=[]
    num_of_points_in_plot=0

    if mode_of_plot == "cubes":
        # в этом случае передали F
        for i in range(len(F)):
            superposition_of_grids_in_reference_area = Grids[i]
            for j in range(len(superposition_of_grids_in_reference_area[0])):
                for k in range(len(superposition_of_grids_in_reference_area[1])):
                    x_1 = superposition_of_grids_in_reference_area[0][j][0]
                    x_2 = superposition_of_grids_in_reference_area[0][j][1]
                    y_1 = superposition_of_grids_in_reference_area[1][k][0]
                    y_2 = superposition_of_grids_in_reference_area[1][k][1]
                    x.append((x_2 + x_1) / 2)
                    y.append((y_2 + y_1) / 2)
                    rects_info.append([x_1, x_2, y_1, y_2])
                    z_otrezok_of_answer = F[i][j][k]
                    z_for_cubes.append([z_otrezok_of_answer[0][0],z_otrezok_of_answer[0][1]])
                    highs.append((z_otrezok_of_answer[0][0] + z_otrezok_of_answer[0][1]) / 2)
                    num_of_points_in_plot += 1
        print("\nnum_of_cubes_in_plot {}".format(num_of_points_in_plot))
    else:
        # в этом случае передали surf
        for i in range(len(F)):
            superposition_of_grids_in_reference_area = Grids[i]
            for j in range(len(superposition_of_grids_in_reference_area[0])):
                for k in range(len(superposition_of_grids_in_reference_area[1])):
                    x_1 = superposition_of_grids_in_reference_area[0][j][0]
                    x_2 = superposition_of_grids_in_reference_area[0][j][1]
                    y_1 = superposition_of_grids_in_reference_area[1][k][0]
                    y_2 = superposition_of_grids_in_reference_area[1][k][1]
                    x.append((x_2 + x_1) / 2)
                    y.append((y_2 + y_1) / 2)
                    rects_info.append([x_1, x_2, y_1, y_2])
                    highs.append(F[i][j][k])
                    num_of_points_in_plot+=1
        print("\nnum_of_points_in_plot {}".format(num_of_points_in_plot))

    if mode_of_plot =="cubes":
        plt.rcParams["figure.figsize"] = [14, 7]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        axs = fig.add_subplot(111, projection='3d')
        norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
        m = cm.ScalarMappable(norm=norm, cmap=cm.jet)

        tmp_len = len(rects_info)
        for i in range(len(rects_info)):
            print("\r interation {} of {}".format(i, tmp_len), end='')
            plot_cube_with_color(axs, rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],z_for_cubes[i][0],z_for_cubes[i][1],
                      m.to_rgba(highs[i]),alpha=0.99)

        axs.set_xlabel(r"$расстояние \:до \:стоп-линии  \: x_{1}, \: м$")
        axs.set_ylabel(r'$скорость \: x_{2}, \: м \cdot c^{-1}$')
        axs.set_zlabel(r'$ускорение \: y_{1}, \: м\cdot c^{-2} $')
        line_of_trajectory=0
        if len(distanses_to_line) !=0:
            line_of_trajectory= axs.scatter(distanses_to_line,v_vec,-6*np.ones(len(distanses_to_line),),c = 'k')
        if len(distanses_to_line) != 0:
            axs.legend([line_of_trajectory], ["траектория точки"])
        plt.colorbar(m, ax=axs)
        axs.set_xlim([0, 361.1])
        axs.set_ylim([0, 22.2])
        axs.set_zlim([-6.0, 6.0])
        axs.azim = 100
        axs.dist = 10
        axs.elev = 20
        plt.show(block=block_canvas)
        print("plot response surface done")

    if mode_of_plot == "map":
        plt.rcParams["figure.figsize"] = [14, 7]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        axs = fig.add_subplot(111)

        # norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
        # cmap = cm.hsv
        # m = cm.ScalarMappable(norm=norm, cmap=cmap)

        # hights = cm.get_cmap('jet', 128)
        # # hights_middle = cm.get_cmap('plasma', 128)
        #
        # newcolors = np.vstack((
        #     # hights_middle(np.linspace(0, 1, 128)),
        #     hights(np.linspace(0, 1, 128)))
        # )
        # newcmp = matplotlib.colors.ListedColormap(newcolors, name='my_cm')
        norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
        m = cm.ScalarMappable(norm=norm, cmap=cm.jet)

        tmp_len = len(rects_info)
        for i in range(len(rects_info)):
            print("\r interation {} of {}".format(i, tmp_len),end='')
            plot_rect(axs, rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],
                      m.to_rgba(highs[i]))

        # plt.xlabel("расстояние до стоп линии, м")
        # plt.ylabel("скорость м/c")
        # plt.title("ускорение м/c^2")
        line_of_trajectory = 0
        if len(distanses_to_line) != 0:
            line_of_trajectory, = axs.plot(distanses_to_line,v_vec,c='k')

        plt.colorbar(m, ax=axs)
        axs.set_xlim([0, 361.1])
        axs.set_ylim([0, 22.2])
        axs.set_xlabel(r"$расстояние \:до \:стоп-линии  \: x_{1}, \: м$")
        axs.set_ylabel(r'$скорость \: x_{2}, \: м \cdot c^{-1}$')
        plt.title(r'$ускорение \: y_{1}, \: м\cdot c^{-2} $')
        if len(distanses_to_line) != 0:
            axs.legend([line_of_trajectory], ["траектория точки"])
        #нарисуем X^k-e
        if plot_rules==True:
            already_checked_x_y_rect = {}
            for i in range(len(list_for_integrate)):
                info = list_for_integrate[i]
                rect=info[0]
                indicies_of_rules = info[1]
                indisies_string= ','.join(str(e) for e in indicies_of_rules)
                a_x=rect[0][0]
                b_x=rect[0][1]
                a_y=rect[1][0]
                b_y=rect[1][1]
                plot_rect_for_rules(axs,a_x,b_x,a_y,b_y,'k')
                # axs.text((a_x+b_x)/2, (a_y+b_y)/2, indisies_string, color="k", fontsize=12)
                # print("rect {} rules {}".format(rect,indicies_of_rules))

        plt.show(block=block_canvas)
        print("plot response surface done")

    if mode_of_plot == "3dmap":
        plt.rcParams["figure.figsize"] = [14, 7]
        plt.rcParams["figure.autolayout"] = True

        fig = plt.figure()
        axs = fig.add_subplot(111, projection='3d')
        # axs.xaxis.label.set_color('red')
        # axs.tick_params(axis='x', colors='red')
        # axs.yaxis.label.set_color('red')
        # axs.tick_params(axis='y', colors='red')
        # axs.zaxis.label.set_color('red')
        # axs.tick_params(axis='z', colors='red')

        # norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
        # cmap = cm.hsv
        # m = cm.ScalarMappable(norm=norm, cmap=cmap)

        norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
        m = cm.ScalarMappable(norm=norm, cmap=cm.jet)

        tmp_len = len(rects_info)
        for i in range(len(rects_info)):
            print("\r interation {} of {}".format(i, tmp_len), end='')
            plot_cube_with_color(axs, rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],highs[i],highs[i],
                      m.to_rgba(highs[i]),alpha=0.99)

        axs.set_xlabel(r"$расстояние \:до \:стоп-линии  \: x_{1}, \: м$")
        axs.set_ylabel(r'$скорость \: x_{2}, \: м \cdot c^{-1}$')
        axs.set_zlabel(r'$ускорение \: y_{1}, \: м\cdot c^{-2} $')
        line_of_trajectory=0
        if len(distanses_to_line) !=0:
            line_of_trajectory= axs.scatter(distanses_to_line,v_vec,-6*np.ones(len(distanses_to_line),),c = 'k')
        if len(distanses_to_line) != 0:
            axs.legend([line_of_trajectory], ["траектория точки"])
        plt.colorbar(m, ax=axs)
        axs.set_xlim([0, 361.1])
        axs.set_ylim([0, 22.2])
        axs.set_zlim([-6.0, 6.0])
        axs.azim = 100
        axs.dist = 10
        axs.elev = 20
        plt.show(block=block_canvas)
        print("plot response surface done")

    if mode_of_plot == "scatter":
        plt.rcParams["figure.figsize"] = [14, 7]
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        axs = fig.add_subplot(111, projection='3d')
        norm = matplotlib.colors.Normalize(vmin=min(highs), vmax=max(highs))
        cmap = cm.jet

        pnt3d1=0
        if len(bayes_areas)==0:
        #     pnt3d1 = axs.scatter(x, y, highs, c='k', alpha=0.99)
        # else:
            pnt3d1 = axs.scatter(x, y, highs, c= highs,cmap=cmap,alpha=0.99)
        # cbar = plt.colorbar(pnt3d)

        axs.set_xlim([0, 361.1])
        axs.set_ylim([0, 22.2])
        axs.set_zlim([-6.0, 6.0])

        line_of_trajectory=0
        if len(distanses_to_line) !=0:
            line_of_trajectory= axs.scatter(distanses_to_line,v_vec,-6*np.ones(len(distanses_to_line),),c = 'k')
        if len(distanses_to_line) != 0:
            axs.legend([line_of_trajectory], ["траектория точки"])
        axs.set_xlabel(r"$расстояние \:до \:стоп-линии  \: x_{1}, \: м$")
        axs.set_ylabel(r'$скорость \: x_{2}, \: м \cdot c^{-1}$')
        axs.set_zlabel(r'$ускорение \: y_{1}, \: м\cdot c^{-2} $')
        if len(problems_areas_info)>0:
            plt.title(r'$V_{k} = \int_{X^k \times Y^k} p(x_{1}|k) p(x_{2}|k)p(y_{1}|k)dx_1dx_2dy_1,color= \sum_{k}V_k,color\in[0,1]$''\n'
                      r'$ \: отн-я\: хар-ка \:проблемности \:области$ ')
        # else:
        #     plt.title( r'$ \: поверхность\: отклика \:контроллера$ ')
        if plot_rules==True:

            for i in range(len(list_for_integrate)):
                info = list_for_integrate[i]
                rect=info[0]
                indicies_of_rules = info[1]
                indisies_string= ','.join(str(e) for e in indicies_of_rules)
                a_x=rect[0][0]
                b_x=rect[0][1]
                a_y=rect[1][0]
                b_y=rect[1][1]
                a_z =rect[2][0]
                b_z= rect[2][1]
                plot_cube_for_rules(axs, a_x, b_x, a_y, b_y, a_z, b_z, "k")
                # axs.text((a_x+b_x)/2, (a_y+b_y)/2,(a_z+b_z)/2, indisies_string, color='k', fontsize=20)
                # print("rect {} rules {}".format(rect,indicies_of_rules))
        if len(problems_areas_info) != 0:
            # max_problem = np.quantile(problems_areas_info,0.25)
            max_problem = np.max(problems_areas_info)
            norm_tmp = matplotlib.colors.Normalize(vmin=0, vmax=1)
            cmap_tmp = cm.PiYG
            m_tmp = cm.ScalarMappable(norm=norm_tmp, cmap=cmap_tmp)
            for i in range(len(list_for_integrate)):
                info = list_for_integrate[i]
                rect=info[0]
                a_x=rect[0][0]
                b_x=rect[0][1]
                a_y=rect[1][0]
                b_y=rect[1][1]
                a_z =rect[2][0]
                b_z= rect[2][1]
                metric_of_problem = problems_areas_info[i]
                if metric_of_problem <= max_problem:
                    plot_cube_with_color(axs, a_x, b_x, a_y, b_y, a_z, b_z, m_tmp.to_rgba(metric_of_problem),alpha=0.3)

        if len(bayes_areas) > 0:
            print("\nplotting bayes areas\n")
            z_bayes = []
            tmp_index= 0
            for i in range(len(bayes_areas)):
                superposition_of_grids_in_reference_area = Grids[i]
                for j in range(len(superposition_of_grids_in_reference_area[0])):
                    for k in range(len(superposition_of_grids_in_reference_area[1])):
                        z_bayes.append([])
                        for index_of_bayes_area in range(len(bayes_areas[i][j][k])):
                            z_bayes[tmp_index].append(bayes_areas[i][j][k][index_of_bayes_area])
                        tmp_index+=1
            tmp_len = len(rects_info)
            num_of_bayes_areas= 0
            m = cm.ScalarMappable(norm=norm, cmap=cm.jet)
            for i in range(len(rects_info)):
                print("\r interation {} of {}".format(i, tmp_len), end='')
                for j in range(len(z_bayes[i])):
                    plot_cube_with_color(axs, rects_info[i][0], rects_info[i][1], rects_info[i][2], rects_info[i][3],
                                         z_bayes[i][j][0], z_bayes[i][j][1],
                                         m.to_rgba((z_bayes[i][j][1]+z_bayes[i][j][0])/2), alpha=0.99)
                    num_of_bayes_areas +=1
            plt.colorbar(m, ax=axs)
            print("\n num_of_bayes_areas {}".format(num_of_bayes_areas))
            #
            # N_steps = len(rects_info)
            # x_for_bayes_area = []
            # y_for_bayes_area = []
            # z_for_bayes_area = []
            # for i in range(len(rects_info)):
            #     print("\r interation {} of {}".format(i, N_steps), end='')
            #     rect = rects_info[i]
            #     a_x = rect[0]
            #     b_x = rect[1]
            #     a_y = rect[2]
            #     b_y = rect[3]
            #     for j in range(len(z_bayes[i])):
            #         a_z = z_bayes[i][j][0]
            #         b_z = z_bayes[i][j][1]
            #         x_for_bayes_area.append((a_x+b_x)/2)
            #         y_for_bayes_area.append((a_y+b_y)/2)
            #         z_for_bayes_area.append((a_z+b_z)/2)
            #         # plot_cube_for_rules(axs, a_x, b_x, a_y, b_y, a_z, b_z, "k")
            # pnt3d3 = axs.scatter(x_for_bayes_area, y_for_bayes_area, z_for_bayes_area, c=z_for_bayes_area, cmap=cmap, alpha=0.4)
            plt.title(r'$ \: квантиль \: = \: $ '+str(quantile_for_bayes_areas)+'\n'
                      r'$   \: черная\: поверхность \:- \:точный\:argmax$''\n'
                r'$ облако\:точек\:-\:разрешенные\:поверхности\:отклика$' )
            # cbar1 = plt.colorbar(pnt3d1)
        if len(bayes_areas)==0:
            cbar1 = plt.colorbar(pnt3d1)
        axs.azim = 100
        axs.dist = 10
        axs.elev = 20
        plt.show(block=block_canvas)
        print("\nplot response surface done")


def find_the_projection_to_X_of_the_superposition_of_grids(new_omega_list, list_for_integrate):

    unique_x_y_areas = {}
    for i in range(len(list_for_integrate)):
        rect = list_for_integrate[i][0]
        x_area_of_rect = rect[0]
        y_area_of_rect = rect[1]
        # проверим, не проверяли ли мы уже эту область в словаре unique_x_y_areass
        already_in_dict = 0
        for key in unique_x_y_areas.keys():
            rect_x_y = unique_x_y_areas[key]
            x_in_rect = rect_x_y[0]
            y_in_rect = rect_x_y[1]
            if x_in_rect[0] == x_area_of_rect[0] and x_in_rect[1] == x_area_of_rect[1] and y_in_rect[0] == \
                    y_area_of_rect[0] and y_in_rect[1] == y_area_of_rect[1]:
                already_in_dict = 1
                break
        if already_in_dict == 1:
            continue
        else:
            unique_x_y_areas.update({i: rect})
    areas_of_x_y = []
    for key in unique_x_y_areas.keys():
        rect =unique_x_y_areas[key]
        areas_of_x_y.append([[rect[0][0], rect[0][1]], [rect[1][0], rect[1][1]]])
    # pprint.pprint(areas_of_x_y)
    # plt.rcParams["figure.figsize"] = [14, 7]
    # plt.rcParams["figure.autolayout"] = True
    # fig = plt.figure()
    # axs = fig.add_subplot(111)
    # for i in range(len(areas_of_x_y)):
    #     rect = areas_of_x_y[i]
    #     plot_rect_for_rules(axs,rect[0][0],rect[0][1],rect[1][0],rect[1][1],'g')
    #
    # axs.set_xlim([-1, 362])
    # axs.set_ylim([-1, 23])
    # axs.set_xlabel(r"$расстояние \:до \:стоп-линии  \: x_{1}, \: м$")
    # axs.set_ylabel(r'$скорость \: x_{2}, \: м \cdot c^{-1}$')
    # plt.title(r'$ускорение \: y_{1}, \: м\cdot c^{-2} $')
    # plt.show(block=True)

    # мы получили все уникальные области по (x,y)
    # теперь нужно пройтись по всем областям вдоль оси z, которые попадают по (x,y) в эти найденные области
    # взять, все эти разбиения по (x,y) и найти их суперпозицию
    output = []
    for i in range(len(areas_of_x_y)):
        x_y_of_reference_area = areas_of_x_y[i]
        x_of_reference_area = x_y_of_reference_area[0]
        y_of_reference_area = x_y_of_reference_area[1]
        # найдем индексы всех областей которые попадат в данный прямоугольник
        areas_that_contain_x_y_area = []
        for j in range(len(list_for_integrate)):
            rect = list_for_integrate[j][0]
            x_of_tmp_area = rect[0]
            y_of_tmp_area = rect[1]
            if x_of_reference_area[0] == x_of_tmp_area[0] and x_of_reference_area[1] == x_of_tmp_area[1] and y_of_reference_area[0] == \
                    y_of_tmp_area[0] and y_of_reference_area[1] == y_of_tmp_area[1]:
                areas_that_contain_x_y_area.append(j)


        grids_of_x = []
        grids_of_y = []
        for index_of_area in areas_that_contain_x_y_area:
            grids_of_x.append(new_omega_list[index_of_area][0])
            grids_of_y.append(new_omega_list[index_of_area][1])
        grid_of_x = []
        for j in range(len(grids_of_x)):
            for k in range(len(grids_of_x[j])):
                grid_of_x.append(grids_of_x[j][k][0])
                grid_of_x.append(grids_of_x[j][k][1])
        superposition_of_x = np.unique(grid_of_x)
        grid_of_y = []
        for j in range(len(grids_of_y)):
            for k in range(len(grids_of_y[j])):
                grid_of_y.append(grids_of_y[j][k][0])
                grid_of_y.append(grids_of_y[j][k][1])
        superposition_of_y = np.unique(grid_of_y)

        superposition_of_grids_in_reference_area = [[],[]]
        for j in range(len(superposition_of_x)-1):
            superposition_of_grids_in_reference_area[0].append([superposition_of_x[j], superposition_of_x[j+1]])
        for j in range(len(superposition_of_y) - 1):
            superposition_of_grids_in_reference_area[1].append([superposition_of_y[j], superposition_of_y[j + 1]])
        output.append([x_y_of_reference_area, areas_that_contain_x_y_area, superposition_of_grids_in_reference_area])
    return output

def copy_p_from_list_of_integrals_to_x_y_list(
        projection_to_x_y_info,
        old_z,old_omega,list_for_integrate, new_omega_list,
        list_of_integrals_z_di):
    # у нас есть проинегрированная плотность(там где это нужно),н-р, из 122 областей, областей где проитегрирована плотность всего
    # 47 штук. но эти 47 штук не пронумерованы, к какой области из 122 они тносятся
    # получим эти индексы
    list_of_indicies_for_indegrated_areas = [] # массив индексов областей, в которых была проинтегрирована плотность
    for i in range(len(list_for_integrate)):
        if len(list_for_integrate[i][1]) > 1:
            list_of_indicies_for_indegrated_areas.append(i)
    list_of_indicies_for_indegrated_areas = np.asarray(list_of_indicies_for_indegrated_areas)
    list_of_z_and_p_with_fixed_x_y = []
    for i in range(len(projection_to_x_y_info)):
        info = projection_to_x_y_info[i]
        x_y_of_reference_area = info[0]
        areas_that_contain_x_y_area = info[1]
        superposition_of_grids_in_reference_area = info[2]
        list_of_z = []
        list_of_p = []

        for j in range(len(superposition_of_grids_in_reference_area[0])):
            list_of_z.append([])
            list_of_p.append([])
            for k in range(len(superposition_of_grids_in_reference_area[1])):

                a_x = superposition_of_grids_in_reference_area[0][j][0]
                b_x = superposition_of_grids_in_reference_area[0][j][1]
                a_y = superposition_of_grids_in_reference_area[1][k][0]
                b_y = superposition_of_grids_in_reference_area[1][k][1]
                input = [(a_x + b_x) / 2, (a_y + b_y) / 2]
                # Зафиксировали (x,y) в этой области и теперь пройдемся вдоль направления z и скопируем данные(само разбиение по z и плотности им соответствующие)
                superposition_of_z_throught_all_areas = []
                all_p_of_z_throught_all_areas = []
                for index_of_area in areas_that_contain_x_y_area:
                    all_rules_in_this_area = list_for_integrate[index_of_area][1]
                    z_in_this_area = new_omega_list[index_of_area][2]
                    p_in_this_area = []
                    if len(all_rules_in_this_area) > 1:
                        # мы попали в проинтегрированную область
                        index_of_integral = np.where(list_of_indicies_for_indegrated_areas == index_of_area)
                        index_of_integral = int(index_of_integral[0])
                        array_of_p = list_of_integrals_z_di[index_of_integral]
                        indicies_of_x_y = get_indicies(input, new_omega_list[index_of_area])
                        p_in_this_area = array_of_p[indicies_of_x_y[0], indicies_of_x_y[1], :]
                    else:
                        # мы попали в непроинтегрированную область
                        p = all_rules_in_this_area[0]
                        array_of_p = old_z[p]
                        a_z = list_for_integrate[index_of_area][0][2][0]
                        b_z = list_for_integrate[index_of_area][0][2][1]
                        start_index_of_z = -1
                        stop_index_of_z = -1

                        for tmp_index in range(len(old_omega[p][2])):
                            otrezok = old_omega[p][2][tmp_index]
                            if a_z >= otrezok[0] and a_z <= otrezok[1]:
                                start_index_of_z = tmp_index
                            if b_z >= otrezok[0] and b_z <= otrezok[1]:
                                stop_index_of_z = tmp_index
                                break



                        index_of_x = -1
                        for tmp_index in range(len(old_omega[p][0])):
                            otrezok = old_omega[p][0][tmp_index]
                            if input[0] >= otrezok[0] and input[0] <= otrezok[1]:
                                index_of_x = tmp_index
                                break
                        index_of_y= -1
                        for tmp_index in range(len(old_omega[p][1])):
                            otrezok = old_omega[p][1][tmp_index]
                            if input[1] >= otrezok[0] and input[1] <= otrezok[1]:
                                index_of_y = tmp_index
                                break

                        p_in_this_area = array_of_p[index_of_x, index_of_y, start_index_of_z:stop_index_of_z+1]

                    #     if index_of_x ==-1 or index_of_y ==-1 or stop_index_of_z==-1 or start_index_of_z==-1:
                    #         print("smth went wrong")
                    #         time.sleep(100000)
                    # if len(z_in_this_area) != len(p_in_this_area):
                    #     print("smth went wrong")
                    #     time.sleep(100000)

                    superposition_of_z_throught_all_areas.append(z_in_this_area)
                    all_p_of_z_throught_all_areas.append(p_in_this_area)
                output_superpos_of_z = []
                output_superpos_of_p = []
                for index_of_chunk in range(len(superposition_of_z_throught_all_areas)):
                    for index_of_otrezok in range(len(superposition_of_z_throught_all_areas[index_of_chunk])):
                        output_superpos_of_z.append(superposition_of_z_throught_all_areas[index_of_chunk][index_of_otrezok])
                        output_superpos_of_p.append(all_p_of_z_throught_all_areas[index_of_chunk][index_of_otrezok])
                list_of_z[j].append(np.asarray(output_superpos_of_z))
                list_of_p[j].append(np.asarray(output_superpos_of_p))

        list_of_z_and_p_with_fixed_x_y.append([list_of_z,list_of_p])
    return list_of_z_and_p_with_fixed_x_y





def plot_p_k(axs, distribution,number_of_rule, number_of_distribution,gradien_under_line):
    axs.grid(True, 'both', 'both')
    x = np.linspace(distribution["a"], distribution["b"], 100)
    x_plot = np.zeros(99, )
    y = np.zeros(99, )
    for i in range(100 - 1):
        y[i] = compute_f(distribution["func"], torch.tensor([x[i], x[i + 1]])).cpu().detach().numpy()
        x_plot[i] = x[i] + (x[i + 1] - x[i]) / 2

    line_1, = axs.plot(x_plot, y)
    color_of_line= line_1.get_color()

        # line_1.set_label('ground truth')
    label_1 = 'k={}'.format(number_of_rule)
    # line_1.set_label()
    if(number_of_distribution==1):
        axs.set_xlabel(r'$x_{1}, \: м$')
        axs.set_ylabel(r'$p_{\xi_{1}|\gamma}(x_{1}|k)$')
    if(number_of_distribution==2):
        axs.set_xlabel(r'$x_{2}, \: м \cdot c^{-1}$')
        axs.set_ylabel(r'$p_{\xi_{2}|\gamma}(x_{2}|k)$')
    if(number_of_distribution==3):
        axs.set_xlabel(r'$ y_{1}, \: м\cdot c^{-2} $')
        axs.set_ylabel(r'$p_{\eta_{1}|\gamma}(y_{1}|k)$')

    return line_1,label_1

def view_p_k(rules,block_c,gradien_under_line):
    figures = []
    axes = []
    labels = {}

    for i in range(3):
        fig, axs = plt.subplots(1, 1)
        figures.append(fig)
        axes.append(axs)

    for i in range(3):

        lines = {}
        for j in range(9):
            labels.update({rules[j+1][i+1]['label']:[]})
        already_plot_indedicies_of_k = {}
        for j in range(9):
            tmp_label = rules[j+1][i+1]['label']
            if tmp_label in already_plot_indedicies_of_k.keys():
                already_plot_indedicies_of_k[tmp_label].append(j+1)
            else:
                already_plot_indedicies_of_k.update({tmp_label:[j+1]})
                line_j, lebel_j = plot_p_k(axes[i], rules[j+1][i+1], j+1, i+1,gradien_under_line)
                lines.update({tmp_label:line_j})
        lines_for_plot= []
        legends_for_plot=[]
        keys=lines.keys()
        for key in keys:
            lines_for_plot.append(lines[key])
            legends_for_plot.append("k= "+','.join(str(e) for e in already_plot_indedicies_of_k[key]))

        axes[i].legend(lines_for_plot, legends_for_plot)
        # plt.show(block=True)

    plt.show(block=block_c)

def make_random_init_values(a_i_j, b_i_j):
    random_seed = {}
    N = 1
    c_i_j_k = np.random.rand(2 * N + 1, 2 * N + 1, 2 * N + 1)
    num_of_rules = 9
    mu_x = np.zeros(num_of_rules, )
    mu_y = np.zeros(num_of_rules, )
    mu_z = np.zeros(num_of_rules, )
    sigma_x =np.zeros(num_of_rules, )
    sigma_y =np.zeros(num_of_rules, )
    sigma_z = np.zeros(num_of_rules, )
    for i in range(num_of_rules):
        mu_x[i] = np.random.uniform(low=b_i_j[i][0] + a_i_j[i][0] * 0.25, high=b_i_j[i][0] + a_i_j[i][0] * 0.75)
        mu_y[i] = np.random.uniform(low=b_i_j[i][1] + a_i_j[i][1] * 0.25, high=b_i_j[i][1] + a_i_j[i][1] * 0.75)
        mu_z[i] = np.random.uniform(low=b_i_j[i][2] + a_i_j[i][2] * 0.25, high=b_i_j[i][2] + a_i_j[i][2] * 0.75)
        sigma_x[i] = np.random.uniform(low=(b_i_j[i][0] - a_i_j[i][0]) * 0.2, high=(b_i_j[i][0] - a_i_j[i][0]) * 0.3)
        sigma_y[i] = np.random.uniform(low=(b_i_j[i][1] - a_i_j[i][1]) * 0.2, high=(b_i_j[i][1] - a_i_j[i][1]) * 0.3)
        sigma_z[i] = np.random.uniform(low=(b_i_j[i][2] - a_i_j[i][2]) * 0.2, high=(b_i_j[i][2] - a_i_j[i][2]) * 0.3)
    random_seed.update({"N": N})
    random_seed.update({"c_i_j_k": c_i_j_k})
    random_seed.update({"mu_x":mu_x})
    random_seed.update({"mu_y": mu_y})
    random_seed.update({"mu_z": mu_z})
    random_seed.update({"sigma_x": sigma_x})
    random_seed.update({"sigma_y": sigma_y})
    random_seed.update({"sigma_z": sigma_z})

    return random_seed

def get_functions_intersect_info_from_indicies(rules,dimension, indicies_of_rules,rect):
    # S_k_i = []
    V_k = []
    S_k_i_in_intersect = []
    V_in_intersect_k = []


    for index_of_rules_index in range(len(indicies_of_rules)):
        # S_k_i.append([])
        rules_index = indicies_of_rules[index_of_rules_index]
        S_k_i_in_intersect.append([])
        for i in range(dimension):
            # S_k_i[k].append(1)
            integral = 0
            if rules[rules_index+1][i+1]["func"]["name"] == "FI":
                integral = quad(integrand_FI, rect[i][0], rect[i][1],
                                args=rules[rules_index+1][i+1]["func"]["params"])[0]
                S_k_i_in_intersect[index_of_rules_index].append(integral)

    # if len(indicies_of_rules)==1:
    #     for i in range(dimension):
    #         V_of_funs *= S_k_i[0][i]
    #         V_in_intersect *= S_k_i_in_intersect[0][i]

    for index_of_rules_index in range(len(indicies_of_rules)):
        V_k_tmp=1
        V_in_intersect_k_tmp=1
        for i in range(dimension):
            # V_k_tmp *= S_k_i[k][i]
            V_in_intersect_k_tmp *= S_k_i_in_intersect[index_of_rules_index][i]
        V_k.append(V_k_tmp)
        V_in_intersect_k.append(V_in_intersect_k_tmp)

    info = 0
    for index_of_rules_index in range(len(indicies_of_rules)):
        info += V_in_intersect_k[index_of_rules_index]/V_k[index_of_rules_index]

    return info



def check_problem_areas(rules,dimension,list_for_integrate):
    problems_info = []
    for i in range(len(list_for_integrate)):
        info = list_for_integrate[i]
        rect = info[0]
        indicies_of_rules = info[1]
        a_x = rect[0][0]
        b_x = rect[0][1]
        a_y = rect[1][0]
        b_y = rect[1][1]
        a_z = rect[2][0]
        b_z = rect[2][1]

        # получим p(x), p(y), p(z) функции и посчитаем площади по ними на этой области опр.
        # хотя на самом деле нужно искать проблемные подобласти-пока будем считать по всей области пересечения целиком
        # и вообще-сам факт пересечения ничего не говорит и по нему нельзя оценить проблемность
        problems_info.append(get_functions_intersect_info_from_indicies(rules,dimension, indicies_of_rules,rect))
    return problems_info

def get_bayes_areas(projection_to_x_y_info, list_of_z_and_p_with_fixed_x_y,quantile_for_bayes_areas):
    start_time = time.time()

    BayesAreas = []
    for i in range(len(projection_to_x_y_info)):
        info = projection_to_x_y_info[i]
        superposition_of_grids_in_reference_area = info[2]
        BayesAreas.append([])
        for j in range(len(superposition_of_grids_in_reference_area[0])):
            BayesAreas[i].append([])
            for k in range(len(superposition_of_grids_in_reference_area[1])):
                all_z = list_of_z_and_p_with_fixed_x_y[i][0][j][k]
                all_p = list_of_z_and_p_with_fixed_x_y[i][1][j][k]
                quantile_value = np.quantile(all_p, quantile_for_bayes_areas)
                tmp_argmax = np.where(all_p >= quantile_value)
                z_of_answers = all_z[tmp_argmax]
                BayesAreas[i][j].append(z_of_answers)

    stop_time= time.time()
    print("time for get bayes areas {}".format(stop_time-start_time))
    return BayesAreas

