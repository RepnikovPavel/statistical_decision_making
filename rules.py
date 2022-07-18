import numpy as np
import torch
from scipy.integrate import quad
from func_determination import funcs

def integrand_FI(x, params):
    return np.exp(-0.5 * np.square(-params[0] + x) / np.square(params[1]))

def recompute_integrals(rules):
    # compute and fill integrals
    for rule in rules:
        for distr in rules[rule]:
            integral = 0
            if rules[rule][distr]["func"]["name"] == "FI":
                integral = quad(integrand_FI, rules[rule][distr]["a"],
                                                           rules[rule][distr]["b"],
                                                           args=rules[rule][distr]["func"]["params"])[0]
            rules[rule][distr]["func"]["norm"] = integral
    return rules
# 'D:\\matlab_installed\\R2020b_full\\bin\\acc_controller_4_9_rules_1.fis
rules_for_acc = {
    1: {
        1: 'm_d',
        2: 'l_v',
        3: 'start_a'
    },
    2: {
        1: 'm_d',
        2: 'm_v',
        3: 'lm_a'
    },
    3: {
        1: 'm_d',
        2: 'h_v',
        3: 'l_a'
    },
    4: {
        1: 'start_stop_d',
        2: 'l_v',
        3: 'l_da'
    },
    5: {
        1: 'start_stop_d',
        2: 'm_v',
        3: 'm_da'
    },
    6: {
        1: 'start_stop_d',
        2: 'h_v',
        3: 'h_da'
    },
    7: {
        1: 'end_stop_d',
        2: 'l_v',
        3: 'l_a'
    },
    8: {
        1: 'end_stop_d',
        2: 'm_v',
        3: 'l_a'
    },
    9: {
        1: 'end_stop_d',
        2: 'h_v',
        3: 'l_a'
    }
}


def make_rules_for_computation(rules):
    rules_for_computation = {}
    for rule in rules:
        rules_for_computation.update({rule:{}})
        for distr in rules[rule]:
            rules_for_computation[rule].update({distr:funcs[rules[rule][distr]]})
            rules_for_computation[rule][distr].update({'label': rules[rule][distr]})
    return rules_for_computation
