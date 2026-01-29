# %%
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic('load_ext', 'autoreload')
ipython.run_line_magic('autoreload', '2')
ipython.run_line_magic('matplotlib', 'ipympl')
# %%
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from os.path import join, dirname
main_dir = dirname(os.getcwd())
sys.path.append(main_dir)

from UTILS import LorenzModel, simulation, latexify

# %%
rng = np.random.default_rng(1234)

# %%
dt = 0.02
model_cont = LorenzModel(dt, measure=[0,1,2], beta=2.)
dims = model_cont["dims"]

theta_star = np.array([10., 30.])

# %%
# Generate data
ntraj = 1
T_per_traj = 50
N_per_traj = int(T_per_traj / dt)
Ucont = np.zeros((N_per_traj, dims["u"]))
Tcont = np.arange(N_per_traj) * dt
ws = np.zeros((N_per_traj, dims["x"]))
vs = np.zeros((N_per_traj, dims["y"]))
list_Ycont = []
for i in range(ntraj):
    x0 = rng.uniform(-10, 10, dims["x"])
    ys_cont  = simulation(model_cont, theta_star, Ucont, ws, vs, x0)
    list_Ycont.append(ys_cont)

# %%
latexify()
i = 0
ys = list_Ycont[i]
fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={'projection': '3d'})

ax.plot(ys[:,0], ys[:,1], ys[:,2], lw=0.5)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$x_3$")
# ax.set_title("Lorenz Attractor")
fig.tight_layout()

# %%
# save Figure
fig.savefig("../FIGURES/Lorenz.pdf", bbox_inches='tight', pad_inches=0.01)

# %%
plt.close("all")

# %%
