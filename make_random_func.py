import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import matplotlib.ticker as mticker

def compute_seq_vec(x_vec, y_vec, z_vec ,c_i_j_k, N, a_1, a_2,a_3, b_1, b_2,b_3):
    l_1 = (b_1 - a_1) / 2
    l_2 = (b_2 - a_2) / 2
    l_3 = (b_3 - a_3) / 2
    x_shape = len(x_vec)
    y_shape = len(y_vec)
    z_shape = len(z_vec)
    alpha_1 = torch.ones(size=(2 * N + 1, x_shape, 1))
    alpha_2 = torch.ones(size=(2 * N + 1, y_shape, 1))
    alpha_3 = torch.ones(size=(2 * N + 1, z_shape, 1))
    for i in range(1, 2 * N + 1, 2):
        alpha_1[i] = torch.cos(i * np.pi * x_vec / l_1)
        alpha_1[i + 1] = torch.sin(i * np.pi * x_vec / l_1)
        alpha_2[i] = torch.cos(i * np.pi * y_vec / l_2)
        alpha_2[i + 1] = torch.sin(i * np.pi * y_vec / l_2)
        alpha_3[i] = torch.cos(i * np.pi * z_vec / l_2)
        alpha_3[i + 1] = torch.sin(i * np.pi * z_vec / l_2)

    # sum = torch.zeros(size=(x_shape, y_shape, z_shape))
    sum = 0
    for i in range(2 * N + 1):
        for j in range(2 * N + 1):
            for k in range(2 * N + 1):
                sum += c_i_j_k[i][j][k] * (alpha_1[i]*torch.t(alpha_2[j])).view(x_shape,y_shape,1)*torch.t(alpha_3[k])

    return torch.abs(sum)

def compute_gauss(x_vec, y_vec,z_vec, mu_x, sigma_x, mu_y, sigma_y,mu_z, sigma_z):
    x_shape = len(x_vec)
    y_shape = len(y_vec)
    return (1 / 2.50662827463 / sigma_x * torch.exp(-1 / 2 * torch.square((x_vec - mu_x) / sigma_x)) * torch.t(1 / 2.50662827463 / sigma_y * torch.exp(-1 / 2 * torch.square((y_vec - mu_y) / sigma_y)))).view(x_shape,y_shape,1)*torch.t(1 / 2.50662827463 / sigma_z * torch.exp(-1 / 2 * torch.square((z_vec - mu_z) / sigma_z)))


def compute_seq_2d(x_vec, y_vec, c_i_j, N, a_1, a_2, b_1, b_2):
    l_1 = (b_1 - a_1) / 2
    l_2 = (b_2 - a_2) / 2
    x_shape = len(x_vec)
    y_shape = len(y_vec)
    alpha_1 = torch.ones(size=(2 * N + 1, x_shape,1))
    alpha_2 = torch.ones(size=(2 * N + 1, y_shape,1))
    for i in range(1, 2 * N + 1, 2):
        alpha_1[i] = torch.cos(i * np.pi * x_vec / l_1)
        alpha_1[i + 1] = torch.sin(i * np.pi * x_vec / l_1)
        alpha_2[i] = torch.cos(i * np.pi * y_vec / l_2)
        alpha_2[i + 1] = torch.sin(i * np.pi * y_vec / l_2)

    sum = 0
    for i in range(2 * N + 1):
        for j in range(2 * N + 1):
            sum += c_i_j[i][j] * (alpha_1[i]*torch.t(alpha_2[j]))

    return torch.abs(sum)

def compute_gauss_2d(x_vec, y_vec, mu_x, sigma_x, mu_y, sigma_y):
    x_shape = len(x_vec)
    y_shape = len(y_vec)
    return (1 / 2.50662827463 / sigma_x * torch.exp(-1 / 2 * torch.square((x_vec - mu_x) / sigma_x)) * torch.t(1 / 2.50662827463 / sigma_y * torch.exp(-1 / 2 * torch.square((y_vec - mu_y) / sigma_y))))

def loss_for_line(x_values,ps_argmax_line,referens_line, medians_to_black_areas,lines_of_ps_argmax):
    # ro_to_ref_line = torch.sum(torch.square(ps_argmax_line-referens_line))
    # ro_to_black_areas = torch.zeros(1,requires_grad=False)
    #
    # ro_to_black_areas =  torch.sum(torch.square(ps_argmax_line-medians_to_black_areas))

    # nepr_loss= torch.zeros(1,requires_grad=False)
    # for i in range(len(x_values)-1):
    #     nepr_loss= nepr_loss +  torch.square(ps_argmax_line[i]-ps_argmax_line[i+1])
    # return ro_to_ref_line*1+ro_to_black_areas*5+nepr_loss*10
    # L_in_target_areas = torch.zeros(1,requires_grad=False)
    # L_out_target_areas = torch.zeros(1, requires_grad=False)
    L_of_line =torch.zeros(1, requires_grad=False)
    for i in range(len(x_values)-1):
        L_of_line = L_of_line +torch.sqrt( torch.square(ps_argmax_line[i+1]-ps_argmax_line[i]) + torch.square(x_values[i+1]-x_values[i]) )
    # return L_out_target_areas/L_in_target_areas
    return L_of_line

def line_optimization(x,y, true_argmax_line, lines_of_ps_argmax):
    ps_argmax_line = torch.tensor(true_argmax_line).clone().detach().requires_grad_(True)
    referens_line = torch.tensor(true_argmax_line).clone().detach().requires_grad_(False)
    x_values= torch.tensor(x).clone().detach().requires_grad_(False)
    optimizer = torch.optim.Adam([ps_argmax_line], 0.01, [0.5, 0.7])
    num_of_steps = 200
    loss_vec= np.zeros(num_of_steps,)
    medians_to_black_areas_tmp = np.zeros(len(x),)
    # Z_with_info= np.zeros(shape=(len(x),len(y)))# характеризует - в данной точке пространтсва есть черная область или нет
    # for i in range(len(x)):
    #     for j in range(len(y)):
    #         if y[j] in lines_of_ps_argmax[i]:
    #             Z_with_info[i][j] = 1
    #         else:
    #             Z_with_info[i][j] = 0
    # a_b_vec = []
    for i in range(len(x)):
        medians_to_black_areas_tmp[i] = np.median(lines_of_ps_argmax[i])
        # нужно определить разрывы
        # a_b_vec.append([np.min(lines_of_ps_argmax[i]),np.min(lines_of_ps_argmax[i])])
        # pr_value= np.min(lines_of_ps_argmax[i])
        # for j in range(len(lines_of_ps_argmax[i])):
        #
        # a_vec.append()
        # найдем разрывы, если они есть
        # a =[]
        # b =[]

        # a = np.min(lines_of_ps_argmax[i])
        # b = np.max(lines_of_ps_argmax[i])
        # center_of_mass_line_tmp[i]=b-a

    medians_to_black_areas = torch.tensor(medians_to_black_areas_tmp,requires_grad=False)
    for i in range(num_of_steps):
        print(i)
        optimizer.zero_grad()
        loss_t = loss_for_line(x_values,ps_argmax_line,referens_line,medians_to_black_areas,lines_of_ps_argmax)
        loss_value = float(loss_t.detach().numpy())
        loss_vec[i]= loss_value
        loss_t.backward()
        optimizer.step()

    fig_loss, axs_loss = plt.subplots(1, 1)
    loss_line, = axs_loss.plot(loss_vec)
    axs_loss.set_title("loss")
    axs_loss.set_yscale("linear")
    return ps_argmax_line.detach().numpy()


a_1 = 0
b_1 = 25
a_2 = 8
b_2 = 16


N_point_x = 1000
N_point_y = 1000
need_log= False
# N = 1
# c_k_p = np.random.rand(2 * N + 1, 2 * N + 1)
# N=20
# for t_i in range(N):
# print(t_i)
fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
ax = fig.add_subplot(1, 1, 1)
plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["figure.autolayout"] = True
mu_1_x = 5
mu_1_y = 11
sigma_1_x = 1
sigma_1_y = 1

mu_2_x= 20
mu_2_y= 14
sigma_2_x = 1
sigma_2_y = 1




x_from_torch = torch.linspace(a_1, b_1, N_point_x).view(N_point_x,1)
y_from_torch = torch.linspace(a_2, b_2, N_point_y).view(N_point_y,1)


X, Y = np.meshgrid(x_from_torch.detach().numpy(), y_from_torch.detach().numpy())

# Z = torch.t(compute_seq_2d(x_from_torch, y_from_torch, c_k_p, N, a_1, a_2, b_1, b_2))
# Z_1  = compute_gauss_2d(x_from_torch,y_from_torch,mu_x+1,sigma_x,mu_y+1,sigma_y)+torch.rand(N_point_x,N_point_y)*0.003
N_1 = 2
N_2 = 2
# c_k_p_1 = np.random.rand(2 * N_1 + 1, 2 * N_1 + 1)
# c_k_p_2 = np.random.rand(2 * N_2 + 1, 2 * N_2 + 1)
filepath_to_make_random_func_dir = "C:\\make_random_func_dir\\"
f1 =filepath_to_make_random_func_dir+"test1.txt"
f2= filepath_to_make_random_func_dir+"test2.txt"
# torch.save(c_k_p_1,f1)
# torch.save(c_k_p_2,f2)
c_k_p_1 = torch.load(f1)
c_k_p_2 =torch.load(f2)
# *compute_seq_2d(x_from_torch, y_from_torch, c_k_p_1, N_2, a_1, a_2, b_1, b_2)
# *compute_seq_2d(x_from_torch, y_from_torch, c_k_p_2, N_1, a_1, a_2, b_1, b_2)
Z_1  = compute_gauss_2d(x_from_torch,y_from_torch,mu_1_x,sigma_1_x,mu_1_y,sigma_1_y)*compute_seq_2d(x_from_torch, y_from_torch, c_k_p_1, N_2, a_1, a_2, b_1, b_2)
Z_2 = compute_gauss_2d(x_from_torch,y_from_torch,mu_2_x,sigma_2_x,mu_2_y,sigma_2_y)*compute_seq_2d(x_from_torch, y_from_torch, c_k_p_2, N_1, a_1, a_2, b_1, b_2)
Z = 0.5*(Z_1+Z_2)

x_for_argmax = x_from_torch.detach().numpy()
# y_for_argmax = np.zeros(len(x_for_argmax),)
z_argmax = np.zeros(len(x_for_argmax),)
y_numpy =y_from_torch.detach().numpy()

Z_surf = torch.t(0.5*(Z_1+Z_2)).detach().numpy()
surf = ax.pcolor(X, Y, Z_surf, cmap=cm.jet,shading='auto')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title(r'$тестовая \: функция \: двух \: аргументов\: u(x,y) $',fontsize=15)
plt.show()

true_argmax= np.zeros(len(x_for_argmax),)
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')
# ax.set_title(r'$тестовая \: функция \: двух \: аргументов\: u(x,y) $')
# fig.colorbar(surf)
# plt.show(block=True)

eq_answs= []
last_line=0
len_tmp_tmp = len(x_for_argmax)
for i in range(len(x_for_argmax)):
    srez = Z[i].detach().numpy()
    maximum = np.max(srez)
    z_argmax[i] =maximum
    quantile = np.quantile(srez, 0.9)
    where_ps_max= np.where(srez > quantile)
    equvivalent_answers = y_numpy[where_ps_max]
    eq_answs.append(equvivalent_answers)
    # y_for_argmax[i] = y_from_torch[argmax_index].detach().numpy()
    true_argmax_tmp = y_numpy[np.where(srez == maximum)] # может выдавать несколько значений тоже-важно
    if len(true_argmax_tmp) > 1:
        print("multiple true argmax")
        print(len(true_argmax_tmp))
        # ax.scatter(x_for_argmax[i]*np.ones(len(true_argmax_tmp)), true_argmax_tmp, maximum*np.ones(len(true_argmax_tmp), ), color='c', markersize=20)
        ax.scatter(x_for_argmax[i]*np.ones(len(true_argmax_tmp)), true_argmax_tmp, color='c', markersize=20)
    else:
        true_argmax[i]= true_argmax_tmp
    plot_x = x_for_argmax[i]*np.ones(len(equvivalent_answers),)
    if i<len_tmp_tmp-1:
    # ax.scatter(plot_x, equvivalent_answers, np.zeros(len(equvivalent_answers),), color='k')
        ax.scatter(plot_x, equvivalent_answers, color='k')
    else:
    #     last_line= ax.scatter(plot_x, equvivalent_answers, np.zeros(len(equvivalent_answers),), color='k',label=r'$argmax^{\alpha}_{y}(p_{\xi,\eta}(x,y))$')
        ax.scatter(plot_x, equvivalent_answers, color='k',label=r'$argmax^{\alpha}_{y}u(x,y)$')

# optimized_argmax = line_optimization(x_for_argmax,y_numpy,true_argmax,eq_answs)

# print(y_for_argmax)\
#
# Z = torch.t(0.5*(Z_1+Z_2)).detach().numpy()
# surf=0
# if need_log==True:
#     surf = ax.plot_surface(X, Y, np.log(Z), cmap=cm.jet,
#                            linewidth=0, antialiased=False, alpha=0.5)
# else:
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.jet,
#                            linewidth=0, antialiased=False, alpha=0.5)

# ax.scatter(x_for_argmax,y_for_argmax,z_argmax,color='k', label=r'$argmax_{y}(p_{\xi,\eta}(x,y))$')
ax.scatter(x_for_argmax,true_argmax,color='r', label=r'$argmax_{y}u(x,y)$')
# ax.scatter(x_for_argmax,optimized_argmax,z_argmax,color='c', label=r'$pseudoargmax_{y}(p_{\xi,\eta}(x,y))$')
# fig.colorbar(surf)
ax.grid(visible=True, which='both', axis='both')
ax.set_xlim([a_1, b_1])
ax.set_ylim([a_2, b_2])

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

# ax.set_zlabel(r'$p_{\xi,\eta}(x,y)$')
ax.legend(fontsize=16)


# ax.azim = 90
# ax.dist = 8
# ax.elev = 90
# ax.set_title(r'$ черные \: области \: (x,y) : p_{\xi,\eta}(x,y) > quantile(p_{\xi,\eta}(x,y),0.9)$',fontsize=16)
plt.show(block=True)



