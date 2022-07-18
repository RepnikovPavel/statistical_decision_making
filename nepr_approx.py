import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time


def compute_FI(x, params):
    return 1 / (params[1] * np.sqrt(2 * np.pi)) * np.exp(-np.square((-params[0] + x) / (np.sqrt(2) * params[1])))


def compute_FI_torch(x, params):
    return 1 / (params[1] * 2.50662827463) * torch.exp(-torch.square((-params[0] + x) / (1.41421356237 * params[1])))


def compute_FI_pr(x, params):
    return compute_FI(x, params) * (params[0] - x) / (np.square(params[1]))


def compute_sigmoid(x, params):
    return 1 / (1 + np.exp(-params[1] * (x - params[0])))


def compute_sigmoid_torch(x, params):
    return 1 / (1 + torch.exp(-params[1] * (x - params[0])))


def compute_sigmoid_pr(x, params):
    return compute_sigmoid(x, params) * (1 - compute_sigmoid(x, params))


# найдем решение системы- совметсное распредление наблюдаемых в среде, действий агнета, номеров правил.
def get_scope_of_definition_of_x_i_k(rules):
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


# функия для вычисления loss
def compute_integral_ot_FI(a, b, mu, sigma):
    return 0.5 * (torch.erf((b - mu) / (sigma * 1.41421356237)) - torch.erf((a - mu) / (sigma * 1.41421356237)))


def compute_FI_torch_mu_sigma(x, mu, sigma):
    return 1 / (sigma * 2.50662827463) * torch.exp(-torch.square((-mu + x) / (1.41421356237 * sigma)))


def take_loss_for_nepr_approx(c_i_j, mu_i_j, sigma_i_j, num_of_rules, dimension, razb_x_i_j, f_ot_x_i_j,
                              scope_of_definition_of_x_i_j, len_of_series, num_of_points, max_squares,vec_with_init_norm,distr_of_rules):
    fi_i_j_p_k = torch.zeros(size=(num_of_rules, dimension, len_of_series, num_of_points), requires_grad=False)
    FI_i_j_p = torch.zeros(size=(num_of_rules, dimension, len_of_series, 1), requires_grad=False)
    A_i_j_k = torch.zeros(size=(num_of_rules, dimension, num_of_points), requires_grad=False)

    for i in range(num_of_rules):
        for j in range(dimension):
            fi_i_j_p_k[i][j] = torch.exp(
                -torch.square((-mu_i_j[i][j] + razb_x_i_j[i][j]) / (1.41421356237 * torch.exp(sigma_i_j[i][j])))) / (
                                       2.50662827463 * torch.exp(sigma_i_j[i][j]))/vec_with_init_norm[i]

    for i in range(num_of_rules):
        for j in range(dimension):
            FI_i_j_p[i][j] = (torch.erf((-mu_i_j[i][j] + scope_of_definition_of_x_i_j[i][j][1]) / (
                    1.41421356237 * torch.exp(sigma_i_j[i][j]))) - torch.erf(
                (-mu_i_j[i][j] + scope_of_definition_of_x_i_j[i][j][0]) / (
                            1.41421356237 * torch.exp(sigma_i_j[i][j])))) * 0.5 / vec_with_init_norm[i]

    D_i_vec = torch.sum(torch.exp(c_i_j) * torch.squeeze(torch.prod(FI_i_j_p, dim=1)), dim=1)
    distr_loss = torch.norm(D_i_vec-distr_of_rules)


    for i in range(num_of_rules):
        A_i_j_k[i][0] = torch.matmul(torch.exp(c_i_j[i]) * torch.t(FI_i_j_p[i][1] * FI_i_j_p[i][2]),
                                     fi_i_j_p_k[i][0]) / D_i_vec[i]
        A_i_j_k[i][1] = torch.matmul(torch.exp(c_i_j[i]) * torch.t(FI_i_j_p[i][0] * FI_i_j_p[i][2]),
                                     fi_i_j_p_k[i][1]) / D_i_vec[i]
        A_i_j_k[i][2] = torch.matmul(torch.exp(c_i_j[i]) * torch.t(FI_i_j_p[i][0] * FI_i_j_p[i][1]),
                                     fi_i_j_p_k[i][2]) / D_i_vec[i]

    norm_loss = torch.square(1 - torch.sum(D_i_vec))
    raznost = f_ot_x_i_j - A_i_j_k

    eq_loss = torch.norm(((raznost*f_ot_x_i_j)/max_squares))

    loss = eq_loss+norm_loss+distr_loss
    return loss, eq_loss, norm_loss, distr_loss


def take_nepr_grad(rules, num_of_rules, dimension, length_of_the_approximating_series):
    len_of_series = length_of_the_approximating_series
    c_i_j_for_norm = torch.log(torch.rand(size=(num_of_rules, len_of_series)))
    # torch.log(torch.rand(size=(num_of_rules, len_of_series))).clone().detach().requires_grad_(True)
    mu_i_j = []
    sigma_i_j = []
    np_mu_i_j = []
    np_sigma_i_j = []
    scope_of_definition_of_x_i_k_for_norm = torch.tensor(get_scope_of_definition_of_x_i_k(rules))
    for i in range(num_of_rules):
        # раскидаем средние по объему в i-м правиле случайным образом. дисперсию инициализируем как sigma = (b-a)/3 + шум \in (b-a)/6
        np_mu_i_j.append(np.zeros(shape=(dimension, len_of_series, 1)))
        np_sigma_i_j.append(np.zeros(shape=(dimension, len_of_series, 1)))
        for j in range(dimension):
            b = scope_of_definition_of_x_i_k_for_norm[i][j][1]
            a = scope_of_definition_of_x_i_k_for_norm[i][j][0]
            for k in range(len_of_series):
                np_mu_i_j[i][j][k] = np.random.uniform(low=a, high=b, size=1)
                np_sigma_i_j[i][j][k] = np.log(
                    np.random.uniform(low=(b - a)* 1/ 3, high=(b - a)*2/3,
                                      size=1))
                 # a +(b-a)/len_of_series*k
                # np_mu_i_j[i][j][k] = (b+a)/2
                # np_sigma_i_j[i][j][k] = np.log((b-a)/3)
    # необходимо отнормировать ряд
    mu_i_j_for_norm = torch.tensor(np_mu_i_j)
    sigma_i_j_for_norm = torch.tensor(np_sigma_i_j)
    FI_i_j_p_for_norm = torch.zeros(size=(num_of_rules, dimension, len_of_series, 1))
    for i in range(num_of_rules):
        for j in range(dimension):
            FI_i_j_p_for_norm[i][j] = (torch.erf((-mu_i_j_for_norm[i][j] + scope_of_definition_of_x_i_k_for_norm[i][j][1]) / (
                    1.41421356237 * torch.exp(sigma_i_j_for_norm[i][j]))) - torch.erf(
                (-mu_i_j_for_norm[i][j] + scope_of_definition_of_x_i_k_for_norm[i][j][0]) / (
                        1.41421356237 * torch.exp(sigma_i_j_for_norm[i][j])))) * 0.5
    vec_with_init_norm_tmp = torch.sum(torch.exp(c_i_j_for_norm) * torch.squeeze(torch.prod(FI_i_j_p_for_norm, dim=1)), dim=1).view(num_of_rules,)
    start_distrib_of_rules = torch.ones(num_of_rules,)/num_of_rules
    vec_with_init_norm = (torch.pow(vec_with_init_norm_tmp, 1/3)*(torch.pow(start_distrib_of_rules,-1/3))).clone().detach().requires_grad_(False)

    distr_of_rules = start_distrib_of_rules.clone().detach().requires_grad_(False)

    c_i_j = c_i_j_for_norm.clone().detach().requires_grad_(True)
    mu_i_j = mu_i_j_for_norm.clone().detach().requires_grad_(True)
    sigma_i_j = sigma_i_j_for_norm.clone().detach().requires_grad_(True)
    scope_of_definition_of_x_i_k = scope_of_definition_of_x_i_k_for_norm.clone().detach().requires_grad_(False)

    tmp_razb_x_i_j = []
    for i in range(num_of_rules):
        tmp_razb_x_i_j.append([])
        for j in range(dimension):
            tmp_razb_x_i_j[i].append(rules[i + 1][j + 1]["razb"])
    razb_x_i_j = []
    for i in range(num_of_rules):
        razb_x_i_j.append([])
        for j in range(dimension):
            razb_x_i_j[i].append([])
            for p in range(len(rules[i + 1][j + 1]["razb"]) - 1):
                razb_x_i_j[i][j].append((tmp_razb_x_i_j[i][j][p + 1] + tmp_razb_x_i_j[i][j][p]) / 2)
    razb_x_i_j = torch.tensor(razb_x_i_j, requires_grad=False)



    num_of_points = len(rules[i + 1][j + 1]["razb"]) - 1
    # coeff_list = []
    # for rule in rules:
    #     coeff_list.append([])
    #     for distribution in rules[rule]:
    #         coeff_list[rule - 1].append((rules[rule][distribution]["b"] - rules[rule][distribution]["a"]))
    # coeff_list = torch.tensor(coeff_list,requires_grad=False).view(9,3,1)

    f_ot_x_i_j = []
    maximus = []
    for i in range(num_of_rules):
        f_ot_x_i_j.append([])
        maximus.append([])
        for j in range(dimension):
            N_p = len(razb_x_i_j[i][j])  # число отрезков
            func_params = rules[i + 1][j + 1]["func"]["params"]
            func_name = rules[i + 1][j + 1]["func"]["name"]
            func_norm = rules[i + 1][j + 1]["func"]["norm"]
            f_ot_x_i_j[i].append([])
            for p in range(N_p):  # номер отрезка
                if func_name == "FI":
                    func_value = compute_FI(x=razb_x_i_j[i][j][p],params=func_params) / func_norm
                    f_ot_x_i_j[i][j].append(func_value)
                if func_name == "sigmoid":
                    func_value= compute_sigmoid(x=razb_x_i_j[i][j][p],params=func_params) / func_norm
                    f_ot_x_i_j[i][j].append(func_value)
            maximus[i].append(max(f_ot_x_i_j[i][j]))
    f_ot_x_i_j = torch.tensor(f_ot_x_i_j,requires_grad=False)
    max_squares = torch.square(torch.tensor(maximus,requires_grad=False)).view(num_of_rules,dimension,1)

    # параметры инициализированы
    train_dict = {"lr": 0.01, "num_of_epochs": 2500,
                  1500: 0.005,
                  # 500: 0.005,
                  }

    optimizer = torch.optim.Adam([c_i_j, mu_i_j, sigma_i_j], train_dict["lr"], [0.5, 0.7])
    loss_vec = np.zeros(train_dict["num_of_epochs"], )
    consyst_vec = np.zeros(train_dict["num_of_epochs"], )
    norm_vec = np.zeros(train_dict["num_of_epochs"], )
    dist_vec = np.zeros(train_dict["num_of_epochs"], )
    last_lr = train_dict["lr"]

    start_time = time.time()
    for i in range(train_dict["num_of_epochs"]):
        optimizer.zero_grad()
        if i in train_dict:
            for g in optimizer.param_groups:
                last_lr = train_dict[i]
                g['lr'] = last_lr

        loss, consistency, norm, distr = take_loss_for_nepr_approx(c_i_j, mu_i_j, sigma_i_j, num_of_rules, dimension,
                                                            razb_x_i_j, f_ot_x_i_j,
                                                            scope_of_definition_of_x_i_k, len_of_series, num_of_points,max_squares,vec_with_init_norm,distr_of_rules)

        loss_vec[i] = loss.cpu().detach().numpy()
        norm_loss = float(norm.cpu().detach().numpy())
        cons_loss = float(consistency.cpu().detach().numpy())
        distr_loss = float(distr.cpu().detach().numpy())
        dist_vec[i] = distr_loss
        consyst_vec[i] = cons_loss
        norm_vec[i] = norm_loss
        print(
            "\r>>   {} ep lr {:10.11f} consistency: {:10.11f}   norm: {:10.11f} distr: {:10.11f}".format(
                i, last_lr,
                cons_loss,
                norm_loss,distr_loss
                ),
            end='')
        loss.backward(retain_graph=False)
        optimizer.step()

    stop_time = time.time()
    print("\ttime per epoch {}".format((stop_time - start_time) / (train_dict["num_of_epochs"] + 1)))

    fig_loss, axs_loss = plt.subplots(1, 3)

    consistency_line, = axs_loss[0].plot(consyst_vec)
    axs_loss[0].set_title("consistency")
    axs_loss[0].set_yscale("linear")

    norm_line, = axs_loss[1].plot(norm_vec)
    axs_loss[1].set_title("norm")
    axs_loss[1].set_yscale("linear")

    distr_line, = axs_loss[2].plot(dist_vec)
    axs_loss[2].set_title("distrib of rules")
    axs_loss[2].set_yscale("linear")

    plt.show(block=True)
    return num_of_rules, dimension, len_of_series, mu_i_j, sigma_i_j, c_i_j,vec_with_init_norm

def compute_intergral_along_ith_axis_on_points_x_vec(points, index_of_distribution, index_of_rule, len_of_series,
                                                     mu_i_j_column, sigma_i_j_column, c_i_p_matrix, a_x, a_y, a_z, b_x,
                                                     b_y, b_z,vec_with_init_norm):
    # input - torch.linspace
    # index_of_distr 0,1,...
    # index_of_rule 0,1,...
    # ouptut consistecy points
    sum_of_series = torch.zeros(len(points), )
    if index_of_distribution == 0:
        for p in range(len_of_series):
            sum_of_series += torch.exp(c_i_p_matrix[index_of_rule][p]) * \
                             compute_integral_ot_FI(a_y, b_y, mu_i_j_column[index_of_rule][1][p], torch.exp(sigma_i_j_column[index_of_rule][1][p])) * \
                             compute_integral_ot_FI(a_z, b_z, mu_i_j_column[index_of_rule][2][p], torch.exp(sigma_i_j_column[index_of_rule][2][p])) * \
                             compute_FI_torch_mu_sigma(points, mu_i_j_column[index_of_rule][0][p], torch.exp(sigma_i_j_column[index_of_rule][0][p]))
    if index_of_distribution == 1:
        for p in range(len_of_series):
            sum_of_series += torch.exp(c_i_p_matrix[index_of_rule][p]) * \
                             compute_integral_ot_FI(a_x, b_x, mu_i_j_column[index_of_rule][0][p], torch.exp(sigma_i_j_column[index_of_rule][0][p])) * \
                             compute_integral_ot_FI(a_z, b_z, mu_i_j_column[index_of_rule][2][p], torch.exp(sigma_i_j_column[index_of_rule][2][p])) * \
                             compute_FI_torch_mu_sigma(points, mu_i_j_column[index_of_rule][1][p], torch.exp(sigma_i_j_column[index_of_rule][1][p]))
    if index_of_distribution == 2:
        for p in range(len_of_series):
            sum_of_series += torch.exp(c_i_p_matrix[index_of_rule][p]) * \
                             compute_integral_ot_FI(a_x, b_x, mu_i_j_column[index_of_rule][0][p], torch.exp(sigma_i_j_column[index_of_rule][0][p])) * \
                             compute_integral_ot_FI(a_y, b_y, mu_i_j_column[index_of_rule][1][p], torch.exp(sigma_i_j_column[index_of_rule][1][p])) * \
                             compute_FI_torch_mu_sigma(points, mu_i_j_column[index_of_rule][2][p], torch.exp(sigma_i_j_column[index_of_rule][2][p]))

    D = 0
    for p in range(len_of_series):
        D += torch.exp(c_i_p_matrix[index_of_rule][p]) * \
             compute_integral_ot_FI(a_x, b_x, mu_i_j_column[index_of_rule][0][p], torch.exp(sigma_i_j_column[index_of_rule][0][p])) * \
             compute_integral_ot_FI(a_y, b_y, mu_i_j_column[index_of_rule][1][p], torch.exp(sigma_i_j_column[index_of_rule][1][p])) * \
             compute_integral_ot_FI(a_z, b_z, mu_i_j_column[index_of_rule][2][p], torch.exp(sigma_i_j_column[index_of_rule][2][p]))
    sum_of_series = sum_of_series / D
    # print(D/(float(vec_with_init_norm[index_of_rule].cpu().detach().numpy())**3))
    return sum_of_series

def plot_f_nepr(axs, rules, distribution, index_of_rule, index_of_distribution, len_of_series, mu, sigma, c, vec_with_init_norm):
    axs.grid(True, 'both', 'both')
    x = np.linspace(distribution["a"], distribution["b"], 100)
    y = np.zeros(100, )
    for i in range(100):
        func_params = distribution["func"]["params"]
        if distribution["func"]["name"] == "FI":
            y[i] = compute_FI(x[i], func_params) / distribution["func"]["norm"]
        if distribution["func"]["name"] == "sigmoid":
            y[i] = compute_sigmoid(x[i], func_params) / distribution["func"]["norm"]
    line_1, = axs.plot(x, y, c='k')
    line_1.set_label('ground truth')

    x_apporx = torch.linspace(distribution["a"], distribution["b"], 200)
    y_approx = compute_intergral_along_ith_axis_on_points_x_vec(points=x_apporx,
                                                                index_of_distribution=index_of_distribution - 1,
                                                                index_of_rule=index_of_rule - 1,
                                                                len_of_series=len_of_series,
                                                                mu_i_j_column=mu,
                                                                sigma_i_j_column=sigma,
                                                                c_i_p_matrix=c,
                                                                a_x=rules[index_of_rule][1]["a"],
                                                                a_y=rules[index_of_rule][2]["a"],
                                                                a_z=rules[index_of_rule][3]["a"],
                                                                b_x=rules[index_of_rule][1]["b"],
                                                                b_y=rules[index_of_rule][2]["b"],
                                                                b_z=rules[index_of_rule][3]["b"],
                                                                vec_with_init_norm=vec_with_init_norm
                                                                )

    line_2, = axs.plot(x_apporx.cpu().detach().numpy(), y_approx.cpu().detach().numpy(),c='b')
    line_2.set_label('approx')

    # x_tmp = torch.linspace(distribution["a"], distribution["b"],30)
    # x_with_target_values = torch.zeros(29,)
    # for i in range(29):
    #     x_with_target_values[i] = (x_tmp[i]+x_tmp[i+1])/2
    # y_with_target_values = compute_intergral_along_ith_axis_on_points_x_vec(points=x_with_target_values,
    #                                                             index_of_distribution=index_of_distribution - 1,
    #                                                             index_of_rule=index_of_rule - 1,
    #                                                             len_of_series=len_of_series,
    #                                                             mu_i_j_column=mu,
    #                                                             sigma_i_j_column=sigma,
    #                                                             c_i_p_matrix=c,
    #                                                             a_x=rules[index_of_rule][1]["a"],
    #                                                             a_y=rules[index_of_rule][2]["a"],
    #                                                             a_z=rules[index_of_rule][3]["a"],
    #                                                             b_x=rules[index_of_rule][1]["b"],
    #                                                             b_y=rules[index_of_rule][2]["b"],
    #                                                             b_z=rules[index_of_rule][3]["b"],
    #                                                             vec_with_init_norm=vec_with_init_norm
    #                                                             )
    #
    # line_3 = axs.scatter(x_with_target_values.cpu().detach().numpy(), y_with_target_values.cpu().detach().numpy())
    # # line_3.set_label('target points')

    axs.set_yscale("linear")
    # axs.set_title("rule: {} distr:{} {}".format(index_of_rule, index_of_distribution,
    #                                             rules[index_of_rule][index_of_distribution]["func"]["name"]))

def plot_consistency_nepr(rules, len_of_series, mu, sigma, c, vec_with_init_norm):
    figures = []
    axes = []
    for i in range(3):
        fig, axs = plt.subplots(1, 1)
        plt.rcParams["figure.figsize"] = [16, 9]
        plt.rcParams["figure.autolayout"] = True

        figures.append(fig)
        axes.append(axs)

        for j in range(9):
            plot_f_nepr(axs, rules, rules[j+1][i+1], j+1, i+1, len_of_series, mu,
                        sigma, c, vec_with_init_norm)

    plt.show(block=True)