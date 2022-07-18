import torch

from nepr_approx import compute_sigmoid
from nepr_approx import compute_FI
from nepr_approx import compute_FI_torch_mu_sigma
from nepr_approx import compute_integral_ot_FI
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import numpy as np


def compute_intergral_along_ith_axis_on_points_x_vec(points, index_of_distribution, index_of_rule, len_of_series,
                                                     c_p_i_j_k, a_x, a_y, a_z, b_x,
                                                     b_y, b_z):
    # input - torch.linspace
    # index_of_distr 0,1,...
    # index_of_rule 0,1,...
    # ouptut consistecy points
    sum_of_series = torch.zeros(len(points), )

    A_p_j_l = torch.zeros(size=(3, 2 * len_of_series + 1), requires_grad=False)
    for j in range(3):
        a_j = 0
        b_j = 0
        if j == 0:
            a_j = a_x
            b_j = b_x
        if j == 1:
            a_j = a_y
            b_j = b_y
        if j == 2:
            a_j = a_z
            b_j = b_z

        l_j = (b_j - a_j) / 2
        A_p_j_l[j][0] = b_j - a_j
        for l in range(1, 2 * len_of_series + 1, 2):
            A_p_j_l[j][l] = (torch.sin(l * np.pi * b_j / l_j) - torch.sin(l * np.pi * a_j / l_j)) / (
                    l * np.pi / l_j)
            A_p_j_l[j][l + 1] = (torch.cos(l * np.pi * a_j / l_j) - torch.cos(l * np.pi * b_j / l_j)) / (
                    l * np.pi / l_j)

    a_p_j_l_k = torch.ones(size=(3, 2 * len_of_series + 1, len(points)), requires_grad=False)

    for j in range(3):
        a_j = 0
        b_j = 0
        if j == 0:
            a_j = a_x
            b_j = b_x
        if j == 1:
            a_j = a_y
            b_j = b_y
        if j == 2:
            a_j = a_z
            b_j = b_z
        l_j = (b_j - a_j) / 2
        for l in range(1, 2 * len_of_series + 1, 2):
            a_p_j_l_k[j][l] = torch.cos(l * np.pi * points / l_j)
            a_p_j_l_k[j][l + 1] = torch.sin(l * np.pi * points / l_j)

    if index_of_distribution == 0:
        for i in range(2 * len_of_series + 1):
            for j in range(2 * len_of_series + 1):
                for k in range(2 * len_of_series + 1):
                    tmp_1 = c_p_i_j_k[index_of_rule][i][j][k]
                    tmp_2 = a_p_j_l_k[0][i]
                    tmp_3 = A_p_j_l[1][j]
                    tmp_4 = A_p_j_l[2][k]
                    sum_of_series += c_p_i_j_k[index_of_rule][i][j][k] * a_p_j_l_k[0][i] * A_p_j_l[1][j] * A_p_j_l[2][k]
    if index_of_distribution == 1:
        for i in range(2 * len_of_series + 1):
            for j in range(2 * len_of_series + 1):
                for k in range(2 * len_of_series + 1):
                    sum_of_series += c_p_i_j_k[index_of_rule][i][j][k] * a_p_j_l_k[1][i] * A_p_j_l[0][j] * A_p_j_l[2][k]
    if index_of_distribution == 2:
        for i in range(2 * len_of_series + 1):
            for j in range(2 * len_of_series + 1):
                for k in range(2 * len_of_series + 1):
                    sum_of_series += c_p_i_j_k[index_of_rule][i][j][k] * a_p_j_l_k[2][i] * A_p_j_l[0][j] * A_p_j_l[1][k]

    D = 0
    for i in range(2 * len_of_series + 1):
        for j in range(2 * len_of_series + 1):
            for k in range(2 * len_of_series + 1):
                D += A_p_j_l[0][i] * A_p_j_l[1][j] * A_p_j_l[2][k] * c_p_i_j_k[index_of_rule][i][j][k]
    sum_of_series = sum_of_series / D

    return sum_of_series


def plot_f_nepr(axs, rules, distribution, index_of_rule, index_of_distribution, len_of_series, c_p_i_j_k):
    axs.grid(True, 'both', 'both')
    x = np.linspace(distribution["a"], distribution["b"], 100)
    y = np.zeros(100, )
    for i in range(100):
        func_params = distribution["func"]["params"]
        if distribution["func"]["name"] == "FI":
            y[i] = compute_FI(x[i], func_params) / distribution["func"]["norm"]
        if distribution["func"]["name"] == "sigmoid":
            y[i] = compute_sigmoid(x[i], func_params) / distribution["func"]["norm"]
    line_1, = axs.plot(x, y)
    line_1.set_label('ground truth')

    x_apporx = torch.linspace(distribution["a"], distribution["b"], 200)
    y_approx = compute_intergral_along_ith_axis_on_points_x_vec(points=x_apporx,
                                                                index_of_distribution=index_of_distribution - 1,
                                                                index_of_rule=index_of_rule - 1,
                                                                len_of_series=len_of_series,
                                                                c_p_i_j_k=c_p_i_j_k,
                                                                a_x=torch.tensor(rules[index_of_rule][1]["a"]),
                                                                a_y=torch.tensor(rules[index_of_rule][2]["a"]),
                                                                a_z=torch.tensor(rules[index_of_rule][3]["a"]),
                                                                b_x=torch.tensor(rules[index_of_rule][1]["b"]),
                                                                b_y=torch.tensor(rules[index_of_rule][2]["b"]),
                                                                b_z=torch.tensor(rules[index_of_rule][3]["b"])

                                                                )

    line_2, = axs.plot(x_apporx.cpu().detach().numpy(), y_approx.cpu().detach().numpy())
    line_2.set_label('approx')

    x_tmp = torch.linspace(distribution["a"], distribution["b"], 30)
    x_with_target_values = torch.zeros(29, )
    for i in range(29):
        x_with_target_values[i] = (x_tmp[i] + x_tmp[i + 1]) / 2
    y_with_target_values = compute_intergral_along_ith_axis_on_points_x_vec(points=x_with_target_values,
                                                                            index_of_distribution=index_of_distribution - 1,
                                                                            index_of_rule=index_of_rule - 1,
                                                                            len_of_series=len_of_series,
                                                                            c_p_i_j_k=c_p_i_j_k,
                                                                            a_x=torch.tensor(
                                                                                rules[index_of_rule][1]["a"]),
                                                                            a_y=torch.tensor(
                                                                                rules[index_of_rule][2]["a"]),
                                                                            a_z=torch.tensor(
                                                                                rules[index_of_rule][3]["a"]),
                                                                            b_x=torch.tensor(
                                                                                rules[index_of_rule][1]["b"]),
                                                                            b_y=torch.tensor(
                                                                                rules[index_of_rule][2]["b"]),
                                                                            b_z=torch.tensor(
                                                                                rules[index_of_rule][3]["b"])
                                                                            )

    line_3 = axs.scatter(x_with_target_values.cpu().detach().numpy(), y_with_target_values.cpu().detach().numpy())
    line_3.set_label('target points')

    axs.set_yscale("log")
    axs.legend()
    axs.set_title("rule: {} distr:{} {}".format(index_of_rule, index_of_distribution,
                                                rules[index_of_rule][index_of_distribution]["func"]["name"]))


def plot_consistency_nepr(rules, len_of_series, c_p_i_j_k):
    figures = []
    axes = []
    for rule in rules:
        fig, axs = plt.subplots(1, 3)
        plt.rcParams["figure.figsize"] = [16, 9]
        plt.rcParams["figure.autolayout"] = True

        figures.append(fig)
        axes.append(axs)
        for distribution in rules[rule]:
            plot_f_nepr(axs[distribution - 1], rules, rules[rule][distribution], rule, distribution, len_of_series,
                        c_p_i_j_k)

    with PdfPages('figures2.pdf') as pdf:
        d = pdf.infodict()
        # d['Title'] = "torch.norm(((raznost*f_ot_x_i_j)/maximus^2))"
        for i in range(len(figures)):
            pdf.savefig(figures[i])

    plt.show(block=True)

# num_of_rules, len_of_series, mu_i_j, sigma_i_j, c_i_j = make_P_from_rules_dim_2(rules, 2, 2, 20)
#
# # построим согласованность
# plot_consistency_nepr(rules,len_of_series, mu_i_j, sigma_i_j, c_i_j)
#
# # построим график самого распредления
#
#
# plt.rcParams["figure.figsize"] = [14, 7]
# plt.rcParams["figure.autolayout"] = True
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = []
# y = []
# z = []
# N_x = 50
# N_y = 50
# x_torch = torch.linspace(start=-5.0, end=5.0, steps=N_x)
# y_torch = torch.linspace(start=-3.0, end=5.0, steps=N_y)
# for i in range(N_x):
#     for j in range(N_y):
#         print("{}%".format((i * N_y + j) / (N_x * N_y) * 100))
#         x.append(float(x_torch[i].cpu().detach().numpy()))
#         y.append(float(y_torch[j].cpu().detach().numpy()))
#         P_integral_po_i = 0
#         for index_of_rule in range(num_of_rules):
#             P_integral_po_i += float(P_func(x_torch[i], y_torch[j], index_of_rule, len_of_series, mu_i_j, sigma_i_j,
#                                             c_i_j).cpu().detach().numpy())
#         z.append(P_integral_po_i)
#
# pnt3d = ax.scatter(x, y, z, c=z, alpha=1)
# plt.xlabel("x")
# plt.ylabel("y")
# cbar = plt.colorbar(pnt3d)
# cbar.set_label("Values (units)")
#
# print("mu {} \\n sigma {} \\n c_i_p {}".format(mu_i_j, sigma_i_j, c_i_j))
#
# # ax.set_xlim([-5.0, 5.0])
# # ax.set_ylim([-5.0, 5.0])
# # ax.set_zlim([0.0, high])
# # path_to_save = "D:\\saved_fig\\" + str(index) + ".png"
# # plt.savefig(path_to_save)
# # # газ и тормоз нажимаются только раздельно поэтому рисует только вдоль осей XZ и YZ
# # # plt.close(fig=fig)
# plt.show()
