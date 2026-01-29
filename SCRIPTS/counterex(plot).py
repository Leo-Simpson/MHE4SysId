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

from UTILS import counterExModel, simulation, parse, latexify
from SOURCE import whole_Method

# %%
rng = np.random.default_rng(123)

# %%
model = counterExModel(Q=1.)
dims = model["dims"]
theta_star = np.array([0.8]) # [a]
theta0 = np.array([0.9]) # [a]

# %%
# Generate data
m = 3
ntraj = 10
N_per_traj = int(1e6)
us = np.zeros((N_per_traj, dims["u"]))
x0 = np.zeros(dims["x"])

US = np.zeros((ntraj, N_per_traj-m, m, dims["u"]))
YS = np.zeros((ntraj, N_per_traj-m, m, dims["y"]))

for i in range(ntraj):
    ws = rng.normal(size=(N_per_traj, dims["x"]))
    vs = rng.normal(size=(N_per_traj, dims["y"]))
    ys = simulation(model, theta_star, us, ws, vs, x0)
    for j in range(m):
        YS[i, :, j, :] = ys[j:-(m-j), :]
# %% 
# Solve problem
Nlist = [int(N) for N in np.geomspace(100, YS.shape[1], num=15)]
nN = len(Nlist)
verbose = 0
estimates_good = {}
estimates_bad = {}
for j, N in enumerate(Nlist):
    print(f"{j+1}/{nN}-th Number of sequences N = {N} ")
    estimates_good[N] = []
    estimates_bad[N] = []
    for i in range(ntraj):
        print(f" re sample {i+1}/{ntraj}")
        list_u = US[i, :N, :, :]
        list_y = YS[i, :N, :, :]
        alpha_good, cost_good, status = whole_Method(
            list_u, list_y, model, theta0, verbose=verbose,
            recompute_cost=False, smart_eta0=False, lti=True,
            arrival_cost=True)
        theta_good = alpha_good[:len(theta0)]
        estimates_good[N].append(theta_good)
        print(f"  good: cost = {cost_good:.4e}, alpha = {alpha_good}")
        alpha_bad, cost_bad, status = whole_Method(
            list_u, list_y, model, theta0, verbose=verbose,
            recompute_cost=False, smart_eta0=False, lti=True,
            arrival_cost=False)
        theta_bad = alpha_bad[:len(theta0)]
        estimates_bad[N].append(theta_bad)
        print(f"  bad : cost = {cost_bad:.4e}, alpha = {alpha_bad}")

# %%
# Prepare lists
all_N = [N for (N,est_list) in estimates_good.items() for theta in est_list]
all_theta_good = [theta for (N,est_list) in estimates_good.items() for theta in est_list]
all_theta_bad = [theta for (N,est_list) in estimates_bad.items() for theta in est_list]

list_max_good = [np.max(estimates_good[N], axis=0) for N in Nlist]
list_min_good = [np.min(estimates_good[N], axis=0) for N in Nlist]
list_max_bad = [np.max(estimates_bad[N], axis=0) for N in Nlist]
list_min_bad = [np.min(estimates_bad[N], axis=0) for N in Nlist]

# %%
color_good = "green"
color_bad = "purple"
alpha = 0.1
markersize=6
latexify()
fig, axs = plt.subplots(len(theta_star), figsize=(6., 3.))
# fig.suptitle(r"Counter example, with $m_e=$"+f"{m-mpred}, "+ r"$m_p=$"+f"{mpred}, ")
# fig.suptitle(r"LTI example, with $m=$"+f"{m}")

for i in range(len(theta_star)):
    true_param = theta_star[i]
    if len(theta_star) == 1:
        ax = axs
    else:
        ax = axs[i]
    # ax.set_title(r"Parameter $\theta_{}$".format(i+1))
    ax.set_xlabel(r"Number of sequences $N$")
    # ax.set_ylabel(r"$\theta_{}$".format(i+1))
    ax.set_ylabel(r"Parameter $\theta$")
    ax.grid(True)
    # ax.set_ylim(0.68,1.02)
    ax.set_xscale("log")
    ax.axhline(true_param, color="black", linestyle="--", label=r"true value $\theta^{\star}$")

    all_theta_i = [theta[i] for theta in all_theta_good]
    list_min_i = [theta[i] for theta in list_min_good]
    list_max_i = [theta[i] for theta in list_max_good]
    # list_mean_i = [theta[i] for theta in list_mean_good]

    ax.plot(all_N, all_theta_i, "x", markersize=4, label=r"$\hat{\theta}$ with constant arrival cost tuning", color=color_good)
    # ax.plot(Nlist, list_mean_i, "-", color=color_good)
    ax.fill_between(Nlist, list_min_i, list_max_i, alpha=alpha, color=color_good)

    all_theta_i = [theta[i] for theta in all_theta_bad]
    list_min_i = [theta[i] for theta in list_min_bad]
    list_max_i = [theta[i] for theta in list_max_bad]
    # list_mean_i = [theta[i] for theta in list_mean_bad]
    ax.plot(all_N, all_theta_i, ".", markersize=markersize, label=r"$\hat{\theta}$ with zero arrival cost", color=color_bad)
    # ax.plot(Nlist, list_mean_i, "-", color=color_bad)
    ax.fill_between(Nlist, list_min_i, list_max_i, alpha=alpha, color=color_bad)
    ax.legend()
fig.tight_layout()
    

# %%
# save Figure
fig.savefig("../FIGURES/CounterExample.pdf", bbox_inches='tight', pad_inches=0.01)

# %%
plt.close("all")

# %%