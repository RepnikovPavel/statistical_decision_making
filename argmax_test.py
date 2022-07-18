import torch
import numpy as np
a_1 = 20
b_1 = 1000
a_2 = 10
b_2 = 100
N_point_x = 100
N_point_y = 100
mu_x= (b_1+a_1)/2
mu_y= (b_2+a_2)/2
sigma_x = (b_1 - a_1) / 4
sigma_y = (b_2 - a_2) / 4

x_from_torch = torch.linspace(a_1, b_1, N_point_x).view(N_point_x,1)
y_from_torch = torch.linspace(a_2, b_2, N_point_y).view(N_point_y,1)


X, Y = np.meshgrid(x_from_torch.detach().numpy(), y_from_torch.detach().numpy())

Z = torch.t(compute_seq_2d(x_from_torch, y_from_torch, c_k_p, N, a_1, a_2, b_1, b_2)).detach().numpy()
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf)

ax.set_xlim([a_1, b_1])
ax.set_ylim([a_2, b_2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()