import numpy as np
import matplotlib.pyplot as plt


def M(R: float) -> float:
    if R < 0:
        return -M(-R)
    else:
        if R < 1000:
            return 0.9 * R
        elif R < 2000:
            return 900 + 700 * (1 - np.exp(-0.006 * (R - 1000)))
        elif R < 6000:
            return 1600
        else:
            return 1600 + 100 * (1 - np.exp(0.001 * (R - 6000)))

def R(R_pr:float,P_old:float,P_new:float,t_after_delata_P:float)->float:

    if R_pr >= 7000 and (P_new-P_old) >= 0:
        return 7000
    return R_pr + (P_new - P_old)*7000*(1 - np.exp(-0.00001*t_after_delata_P))

N = 20000
t_vec = np.linspace(0, 15, N)
delta_t = t_vec[1]-t_vec[0]
R_vec = 1000*np.ones(N,)

delta_t_for_alg = 1
x_vec_for_P = np.zeros(int(delta_t_for_alg/delta_t),)
P_vec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
t_vec_for_P = np.zeros(len(P_vec),)
t_vec_for_P[0] = 0

j = 0
k = 0
index_frec = int(delta_t_for_alg/delta_t)
P_new = P_vec[0]
P_old = P_vec[0]
R_old = R_vec[0]
index_from_new_P = 0
for i in range(1, N, 1):
    j += 1
    if np.mod(j, index_frec) == 0:
        k += 1
        P_old = P_new
        P_new = P_vec[k]
        t_vec_for_P[k] = t_vec_for_P[k - 1] + delta_t * (i - index_from_new_P)
        R_old = R_vec[i]
        index_from_new_P = i

    R_vec[i] = R(R_old, P_old=P_old, P_new=P_new, t_after_delata_P=delta_t*(i-index_from_new_P))

fig,ax = plt.subplots(1,2)
ax[0].plot(t_vec_for_P[:k],P_vec[:k])
ax[1].plot(t_vec, R_vec)
plt.show()


# N = 200
# R_vec = np.linspace(-7000, 7000, N)
# M_vec = np.zeros(N, )
# for i in range(N):
#     M_vec[i] = M(R_vec[i])
# plt.plot(R_vec, M_vec)
#
# plt.show()
