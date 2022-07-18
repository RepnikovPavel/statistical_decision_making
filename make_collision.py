import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

scope_of_definition = [[0, 10], [0, 22]]

ax1_1 = 1
bx1_1 = 5
ax1_2 = 2
bx1_2 = 7

ax2_1 = 2
bx2_1 = 3
ax2_2 = 4
bx2_2 = 6

ax3_1 = 6
bx3_1 = 7
ax3_2 = 8
bx3_2 = 19

ax4_1 = 5
bx4_1 = 8
ax4_2 = 5
bx4_2 = 20

list_of_rect = [
    [[ax1_1, bx1_1], [ax1_2, bx1_2]],
    [[ax2_1, bx2_1], [ax2_2, bx2_2]],
    [[ax3_1, bx3_1], [ax3_2, bx3_2]],
    [[ax4_1, bx4_1], [ax4_2, bx4_2]],
]
omega = [[], []]
# для начала найдем левую границу
# фиксируем ось
x_left_sides = []
y_left_sides = []
x_right_sides = []
y_right_sides = []
for i in range(len(list_of_rect)):
    x_left_sides.append(list_of_rect[i][0][0])
    y_left_sides.append(list_of_rect[i][1][0])
    x_right_sides.append(list_of_rect[i][0][1])
    y_right_sides.append(list_of_rect[i][1][1])
x_left_sides = np.asarray(x_left_sides)
y_left_sides = np.asarray(y_left_sides)
x_right_sides = np.asarray(x_right_sides)
y_right_sides = np.asarray(y_right_sides)

mins_of_x_i = [np.min(x_left_sides), np.min(y_left_sides)]
maxs_of_x_i = [np.max(x_right_sides), np.max(y_right_sides)]
new_omega = []
# кадой риске в new_omega будет соответствовать список прямоугольников, которые ее спродуцировали

# numbers_of_rect_in_omega = []
# пропустили нулевую итерацию( может тас случиться что нулевую левую границу спродуцировало много прямоугольников)
for i in range(2):
    new_omega.append([mins_of_x_i[i]])

# numbers_of_rect_in_omega.append(list(np.where(x_left_sides == x_left_sides.min())))
# numbers_of_rect_in_omega.append(list(np.where(y_left_sides == y_left_sides.min())))

for i in range(2):
    while True:
        tmp_left_border = new_omega[i][-1]
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
                    # узнаем какой прямоугольник сделал риску на оси
                    # args_of_tmp_min_new_left_border.append(p)
                    break
        tmp = np.asarray(tmp_min_new_left_border)
        tmp_min = np.min(tmp)
        new_omega[i].append(tmp_min)
        # найдем позиции на которых встретилось минимальное значение
        # и по этим позициям запишем- какие прямоугольники соответствуют этому минимуму
        # args_of_tmp_min_new_left_border = np.asarray(args_of_tmp_min_new_left_border)
        # numbers_of_rect_in_omega[i].append(args_of_tmp_min_new_left_border[np.where(tmp == tmp.min())])

return_new_omega = []
for i in range(len(mins_of_x_i)):
    return_new_omega.append([])
    for j in range(len(new_omega[i]) - 1):
        return_new_omega[i].append([new_omega[i][j], new_omega[i][j + 1]])
# return_numbers_of_rects_in_omega = []
# for i in range(len(mins_of_x_i)):
#     return_numbers_of_rects_in_omega.append([])
#     for j in range(len(new_omega[i]) - 1):
#         return_numbers_of_rects_in_omega[i].append([numbers_of_rect_in_omega[i][j], numbers_of_rect_in_omega[i][j + 1]])
# т.е. получено вдоль каждой оси- список отрезков - массив new_omega -1й индекс номер оси - 2й индекс номер отрезка и
# получено вдоль каждой оси- список соответствующий отрезкам, каждый из этих списков соответствует отрезку с таким эе
# индексом этот список хранит два списка - первый список-какие прямоугольники спродуцировали левую сторону отрезка
# второй список -какие прямоугольники спродуцировали правую сторону отрезка

# в итоге новые прямоугольники будут храниться так: будут храниться один список return_new_omega и тензор размерности
# (shape_new_omega[0],shape_new_omega[1],...,shape_new_omega[8], info_size) который хранит информацию о пересечении
# или принадлежности какому-то правилу. при этом return_numbers_of_rects_in_omega хранить потом не нужно,
# т.к. он нужет только для построения тезора с информацией
tensor_with_information = []
for i in range(len(return_new_omega[0])):
    tensor_with_information.append([])
    for j in range(len(return_new_omega[1])):
        tensor_with_information[i].append([])
# проходимся по всем прямоугольникам, которые являются декартовым произедением отрезков
for i in range(len(return_new_omega[0])):
    for j in range(len(return_new_omega[1])):
        # # i,j прямоугольник  iй отрезок на x_1 , jй отрезок на x_2
        # # узнаем, какие правила, участвовали с создании этого прямоугольника
        # list_of_rules_in_this_rect_from_rect_axis_tick_information = np.unique(
        #     #                              индекс_оси  индекс_отрезка индекс_стороны_отрезка
        #     np.concatenate((return_numbers_of_rects_in_omega[0][i][0], return_numbers_of_rects_in_omega[0][i][1],
        #                     return_numbers_of_rects_in_omega[1][j][0], return_numbers_of_rects_in_omega[1][j][1])))
        # # узнаем в каких правилах(из последнего списка) прямоугольник лежит целиком
        # list_of_rules_in_this_rect =[]
        # for index_of_rect in list_of_rules_in_this_rect_from_rect_axis_tick_information:

        #     просто для фиксированного прямоугольника проверяем на вложенность во все ослальные прямоугольники
        #     соответствующие правилам
        segment_0 = return_new_omega[0][i]
        segment_1 = return_new_omega[1][j]
        # segment_2 = return_new_omega[2][k]
        # и т.д.
        # tensor_with_information[i][j].append([])
        for index_of_rect in range(len(list_of_rect)):
            if (segment_0[0] >= list_of_rect[index_of_rect][0][0] and segment_0[1] <= list_of_rect[index_of_rect][0][1]) == 0:
                continue
            if (segment_1[0] >= list_of_rect[index_of_rect][1][0] and segment_1[1] <= list_of_rect[index_of_rect][1][1]) == 0:
                continue
        #   if (segment_2[0] >= list_of_rect[index_of_rect][2][0] and segment_2[1] <= list_of_rect[index_of_rect][2][1]) == 0:
        #       continue
        #   и т.д.
            tensor_with_information[i][j].append(index_of_rect)
        # # теперь нужно проверить, не находится ли данный прямоугольник внутри других правил,которые не были в первом списке
        # # т.к. отметки на осях такое событие могут не поймать
        # list_of_rules_in_this_rect_from_all_axis_tick_information =
        # # итоговая информация об этом прямоугольнике будет объединением двух последних числовых множеств- не списков.
        # union =
        # tensor_with_information[i][j]= union

import pprint
pprint.pprint(tensor_with_information)

# # list_of_rect = [
# #     [[ax1_1, bx1_1], [ax1_2, bx1_2]],
# #     [[ax2_1, bx2_1], [ax2_2, bx2_2]],
# # ]
# new_list_of_rects = np.zeros(shape=(len(return_new_omega[0]), len(return_new_omega[1]), 4))
# for i in range(len(return_new_omega[0])):
#     for j in range(len(return_new_omega[1])):
#         # добавим левую и правую точку по первой оси и по второй оси
#         new_list_of_rects[i][j][0] = return_new_omega[0][i][0]
#         new_list_of_rects[i][j][1] = return_new_omega[0][i][1]
#         new_list_of_rects[i][j][2] = return_new_omega[1][j][0]
#         new_list_of_rects[i][j][3] = return_new_omega[1][j][1]
# # define Matplotlib figure and axis
# fig, ax = plt.subplots()
#
# # create simple line plot
# ax.plot([-1, -1], [0, 10])
# ax.set_xlim([0, 10])
# ax.set_ylim([0, 22])
# plt.xticks(np.arange(0, 10, step=1))
# plt.yticks(np.arange(0, 22, step=1))
# plt.grid(which='both', axis='both')
# # add rectangle to plot
# for i in range(len(list_of_rect)):
#     ax.add_patch(
#         Rectangle((list_of_rect[i][0][0], list_of_rect[i][1][0]), list_of_rect[i][0][1] - list_of_rect[i][0][0],
#                   list_of_rect[i][1][1] - list_of_rect[i][1][0],
#                   edgecolor='red',
#                   facecolor='blue',
#                   fill=False,
#                   lw=5))
# ax.set_title("before")
#
# fig2, ax2 = plt.subplots()
# plt.xticks(np.arange(0, 10, step=1))
# plt.yticks(np.arange(0, 22, step=1))
# plt.grid(which='both', axis='both')
# # add rectangle to plot
# shape = np.shape(new_list_of_rects)
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         ax2.add_patch(
#             Rectangle((new_list_of_rects[i][j][0], new_list_of_rects[i][j][2]),
#                       new_list_of_rects[i][j][1] - new_list_of_rects[i][j][0],
#                       new_list_of_rects[i][j][3] - new_list_of_rects[i][j][2],
#                       edgecolor='red',
#                       facecolor='blue',
#                       fill=False,
#                       lw=5))
# ax.set_xlim([0, 10])
# ax.set_ylim([0, 22])
# ax2.set_title("after")
# # display plot
# plt.show()
