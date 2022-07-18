from nepr_approx import take_nepr_grad
from nepr_approx import plot_consistency_nepr
import torch
from execute_functions import fill_ranv_razb
from rules import rules_for_acc, make_rules_for_computation, recompute_integrals

init_rules = rules_for_acc
rules_for_computation = make_rules_for_computation(init_rules)
rules_for_computation = recompute_integrals(rules_for_computation)
rules_for_computation = fill_ranv_razb(rules_for_computation, fill_same_N=True, same_N=40)
rules = rules_for_computation
num_of_rules, dimension, len_of_series, mu_i_j, sigma_i_j, c_i_j,vec_with_init_norm = take_nepr_grad(rules, 9, 3,50)
plot_consistency_nepr(rules, 50,mu_i_j, sigma_i_j, c_i_j, vec_with_init_norm)




