# последний рабочий проект на UE4 который работает с этой программой: test_0

import pprint

import shutil
import numpy as np
import time

from execute_functions import take_gradients
from execute_functions import make_omega_and_np_omega
from execute_functions import integrate_z_di
from support_functions import clear_dir_for_chat_with_sim, get_bayes_areas, \
    find_the_projection_to_X_of_the_superposition_of_grids, copy_p_from_list_of_integrals_to_x_y_list, \
    compute_F, compute_lip, make_response_surface, make_response_surf_with_grad
from rules import recompute_integrals
from support_functions import compute_a_h_f_coeff_list
from support_functions import make_list_of_rect_from_rules
from support_functions import make_tensor_from_list_of_rects
from support_functions import make_list_for_integrate
from support_functions import eval_list_of_integrals
from support_functions import torch_to_numpy_z
from execute_functions import check_and_plot_p
from execute_functions import chat_with_sim
from support_functions import make_init_z
from execute_functions import fill_ranv_razb
from support_functions import make_RO_tensor
from execute_functions import take_multiple_gradients_from_one_init_point
from execute_functions import chat_with_sim_one_contoller

from support_functions import make_omega_for_init_z
from support_functions import plot_multiple_z_after_training_and_init_z
from support_functions import make_distrib_of_rules
from support_functions import plot_response_surface
from support_functions import view_p_k

from execute_functions import check_and_plot_muptiple_p

from support_functions import make_new_omega_list

from trivia_simulation import simulate_one_car
# from support_functions import plot_max_surface

from rules import make_rules_for_computation
from execute_functions import train_razb_potoch_approx
import torch
from support_functions import make_random_init_values
from support_functions import check_problem_areas

base_path = "D:\\saved_tensors_81"
filename_for_messaging_with_engine = "C:\\for_engine_commands\\to_unreal.txt"

num_of_experiment = 4
#mode_of_program = "training_rules"
mode_of_program = "traing_distrib"
# mode_of_program = "traing_razb"
mode_of_approximation = "pointwise_approximation"

clear_dir_for_chat_with_sim(path_to_chating="C:\\for_engine_commands")

if mode_of_program == "traing_razb":
    # rules_filename = base_path + "\\" + "rules_exp2_epoch2_element1.txt"
    # rules = torch.load(rules_filename)
    # pprint.pprint(rules)
    # rules = rules_for_evolution_by_distib
    # recompute_integrals(rules)
    # # filled_rules_from_razb_filename = train_razb_potoch_approx(rules)
    # filled_rules_from_razb_filename = fill_ranv_razb(rules)
    filepath_to_rules = "C:\\writed_rules\\rules_for_acc.txt"
    from rules import rules_for_acc
    init_rules = rules_for_acc
    rules_for_computation = make_rules_for_computation(init_rules)
    rules_for_computation = recompute_integrals(rules_for_computation)
    rules_for_computation = fill_ranv_razb(rules_for_computation,fill_same_N = True,same_N=40)
    torch.save(rules_for_computation, filepath_to_rules)
    print("done")


if mode_of_program == "traing_distrib":
    # filepath_to_rules = "C:\\writed_rules\\rules_for_acc.txt"
    # from rules import rules_for_acc
    # init_rules = rules_for_acc
    # rules_for_computation = make_rules_for_computation(init_rules)
    # rules_for_computation = recompute_integrals(rules_for_computation)
    # rules_for_computation = fill_ranv_razb(rules_for_computation,fill_same_N = True,same_N=20)
    # torch.save(rules_for_computation, filepath_to_rules)
    rules_filename = "C:\\writed_rules\\rules_for_acc.txt"
    rules = torch.load(rules_filename)
    #
    view_p_k(rules, block_c=True,gradien_under_line = True)
    time.sleep(10000)
    # #
    # omega, np_omega = make_omega_and_np_omega(rules=rules)
    # a, h, f, coeff_list = compute_a_h_f_coeff_list(rules=rules, omega=omega)
    #
    list_of_rect = make_list_of_rect_from_rules(rules=rules)
    # tensor_with_info, omega_for_rules = make_tensor_from_list_of_rects(list_of_rect=list_of_rect)
    # list_for_integrate = make_list_for_integrate(omega_for_rules=omega_for_rules, tensor_with_info=tensor_with_info)
    # new_omega_list = make_new_omega_list(np_omega=np_omega, list_for_integrate=list_for_integrate,
    #                                       print_num_of_rect=True)
    #
    # problems_areas_info = check_problem_areas(rules, 3, list_for_integrate)
    # start_time = time.time()
    # previous_z_lists = []
    # new_blood_z_list = []
    # new_blood_list_of_integrals = []
    # distrib_of_rules = np.ones(9, ) / 9
    #
    # a_i_j, b_i_j, x_i_vec, y_i_vec, z_i_vec = make_omega_for_init_z(rules=rules, num_of_rules=9, dimension=3)
    # filepath_to_saved_init_z_random_seed = "C:\\saved_random_seed\\random_seed_1.txt"
    # # random_seed = make_random_init_values(a_i_j=a_i_j, b_i_j=b_i_j)
    # # torch.save(random_seed, filepath_to_saved_init_z_random_seed)
    # random_seed = torch.load(filepath_to_saved_init_z_random_seed)
    #
    # z_init = make_init_z(num_of_rules=9, distrib_of_rules=distrib_of_rules, dimension=3, a_i_j=a_i_j, b_i_j=b_i_j,random_seed=random_seed,
    #                      x_i_vec=x_i_vec, y_i_vec=y_i_vec,
    #                      z_i_vec=z_i_vec, razb_tensor_a=a, list_of_rect=list_of_rect)
    #
    # train_dict = {"max_num_of_epochs": 2000, "lr": 0.01}
    # z_list = take_gradients(
    #     num_of_rules=9,
    #     dimension=3,
    #     rules=rules,
    #     omega=omega,
    #     train_dict=train_dict,
    #     a=a, h=h, f=f, coeff_list=coeff_list,
    #     init_z=z_init, distrib_of_rules=distrib_of_rules,
    #     check_target_values=False,
    #     min_cons = 0.001,
    #     min_norm = 0.001,
    #     min_distr = 0.001,
    #     previous_z_lists=[],
    #     problems_areas_info=problems_areas_info,
    #     list_for_integrate=list_for_integrate,
    #     old_omega=np_omega,
    #     print_time_of_this_func=True,
    #     plot_gradien_loss=True,
    #     plot_consystency=True,
    #     print_num_of_restart_gradient=False,
    #     print_tmp_cons_and_loss=True)
    #
    # list_of_integrals_z_di = integrate_z_di(
    #     z=z_list,
    #     np_omega=np_omega,
    #     new_omega_list=new_omega_list,
    #     list_for_integrate=list_for_integrate,
    #     print_time_of_this_func=True,
    #     print_time_of_integration_for_each_rect=False,print_tmp_computed=False)
    #
    # z_np = torch_to_numpy_z(z_list)
    #
    file_to_car = "C:\\saved_cars\\car_5\\"
    file_to_z_np = file_to_car + "file_to_z_np" + ".txt"
    file_to_omega_np = file_to_car + "file_to_omega_np" + ".txt"
    file_to_list_of_integrate = file_to_car + "file_to_list_of_integrate" + ".txt"
    file_to_new_omega = file_to_car + "file_to_new_omega" + ".txt"
    file_to_list_of_integrals = file_to_car + "file_to_list_of_integrals" + ".txt"
    #
    # torch.save(z_np,file_to_z_np)
    # torch.save(np_omega,file_to_omega_np)
    # torch.save(list_for_integrate,file_to_list_of_integrate)
    # torch.save(new_omega_list,file_to_new_omega)
    # torch.save(list_of_integrals_z_di,file_to_list_of_integrals)

    z_np = torch.load(file_to_z_np)
    np_omega = torch.load(file_to_omega_np)
    list_for_integrate = torch.load(file_to_list_of_integrate)
    new_omega_list = torch.load(file_to_new_omega)
    list_of_integrals_z_di = torch.load(file_to_list_of_integrals)
    # problems_areas_info = check_problem_areas(rules, 3, list_for_integrate)
    #
    # distanses_to_line, v_vec = simulate_one_car(old_z=z_np, old_omega=np_omega, list_for_integrate=list_for_integrate,
    #                   new_omega_list=new_omega_list, list_of_integrals_z_di=list_of_integrals_z_di,plot_simul =False,block_canvas =False)
    projection_to_x_y_info = find_the_projection_to_X_of_the_superposition_of_grids(new_omega_list, list_for_integrate)
    list_of_z_and_p_with_fixed_x_y =copy_p_from_list_of_integrals_to_x_y_list(projection_to_x_y_info,
        z_np,np_omega, list_for_integrate, new_omega_list,
        list_of_integrals_z_di)
    quantile_for_bayes_areas = 0.9
    BayesAreas = get_bayes_areas(projection_to_x_y_info, list_of_z_and_p_with_fixed_x_y,quantile_for_bayes_areas)
    F, Rects, Grids = compute_F(projection_to_x_y_info, list_of_z_and_p_with_fixed_x_y)
    # surf = make_response_surface(F, Grids,"grad")
    surf = make_response_surf_with_grad(F, Grids, Rects, BayesAreas)

    x_rects_for_lips,y_rects_for_lips,lips_values = compute_lip(surf, Rects, Grids, U= [5, 0.25],X_Y_Z_rects=list_of_rect)
    # "cubes", "",F
    # "3dmap","",surf
    plot_response_surface("3dmap", "",surf, Rects, Grids,list_for_integrate,
                           block_canvas=True, distanses_to_line=[], v_vec=[], plot_rules=False, problems_areas_info=[], bayes_areas=[],quantile_for_bayes_areas=quantile_for_bayes_areas,
                          plot_lip=True,x_rects_for_lips=x_rects_for_lips,y_rects_for_lips=y_rects_for_lips,lips_values=lips_values)

    # # plot_max_surface("scatter", "", z_np, np_omega, list_for_integrate, new_omega_list, list_of_integrals_z_di,
    # #                        block_canvas=True,distanses_to_line=distanses_to_line, v_vec=v_vec)
