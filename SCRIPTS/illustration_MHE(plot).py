# %%
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic('load_ext', 'autoreload')
ipython.run_line_magic('autoreload', '2')
ipython.run_line_magic('matplotlib', 'ipympl')
# %%
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import sys, os
from os.path import join, dirname
main_dir = dirname(os.getcwd())
sys.path.append(main_dir)

from UTILS import OscillatorModel, simulation, parse, subsequences, mhe_plot, latexify
from SOURCE import simple_MHE

# %%
rng = np.random.default_rng(1234)

# %%
dt = 1e-2
sampling_time = 1.
jump = int(sampling_time / dt)
model = OscillatorModel(dt)
dims = model["dims"]
theta_star = np.array([1]) # [omega]

# %%
# Generate data
ntraj = 3
T_per_traj = 25
N_per_traj = int(T_per_traj / dt)
Q_scale = 0.
R_scale = 1.
Ucont = np.zeros((N_per_traj, dims["u"]))
Tcont = np.arange(N_per_traj) * dt
list_Ycont = []
list_Y, list_T = [], []
for i in range(ntraj):
    x0 = rng.uniform(-10, 10, dims["x"])
    ws = rng.normal(size=(N_per_traj, dims["x"])) * Q_scale
    vs = rng.normal(size=(N_per_traj, dims["y"])) * R_scale
    ys_cont  = simulation(model, theta_star, Ucont, ws, 0*vs, x0)
    ys = simulation(model, theta_star, Ucont, ws, vs, x0)
    list_Ycont.append(ys_cont)
    list_Y.append(parse(ys, jump))
    list_T.append(parse(Tcont, jump))

# %%
time_window = 5.
m = int(time_window / (dt*jump))
n_per_traj = 3
list_subT, list_subY = subsequences(list_T, list_Y, m, n_per_traj)

# %%
# Compute MHE
theta_est = theta_star *0.7
u0 = np.zeros(1)
x = ca.SX.sym("x", dims["x"])
casadi_fmodel = ca.Function("fmodel", [x], [model["f"](x, u0, theta_est)])
casadi_gmodel = ca.Function("gmodel", [x], [model["g"](x, theta_est)])
Sigmainv = np.eye(dims["x"])
x0bar = 0 * np.ones(dims["x"])
y0bar = model["g"](x0bar, theta_est)
Qinv = model["Qinv"](theta_est)
Rinv = model["Rinv"](theta_est)
scale_Qmhe = 100.
scale_Rmhe = 100.

list_subhaty, list_subhaty_t = [], []
for subT,subY in zip(list_subT, list_subY):
    subhaty, subhaty_t = [], []
    for T, Y in zip(subT, subY):
        yest = Y[:-1]
        mest = len(yest)
        ks = np.arange(1, mest+1, dtype=int) * jump
        horizon = (1+mest + 1) * jump
        y_hat, _ = simple_MHE(casadi_fmodel, casadi_gmodel,
                           horizon, ks, yest,
                           x0bar, Sigmainv, Qinv / scale_Qmhe, Rinv / scale_Rmhe)
        t_y_hat = T[0] + np.arange(len(y_hat)) * dt - jump*dt
        
        subhaty.append(y_hat)
        subhaty_t.append(t_y_hat)

    list_subhaty.append(subhaty)
    list_subhaty_t.append(subhaty_t)

# %%
# Plot data
i = 0 # rng.integers(0, ntraj)
Ycont = list_Ycont[i]
subT = list_subT[i]
subY = list_subY[i]
subhaty = list_subhaty[i]
subhaty_t = list_subhaty_t[i]
last_points = [(seqT[-1], seqY[-1]) for (seqT, seqY) in zip(subT, subY)]

# %%
latexify()
fig = mhe_plot(subT, subY, Tcont, Ycont,
               xticks=
               np.arange(0, T_per_traj+1, 5, dtype=int),
               mhe=(subhaty, subhaty_t, last_points, y0bar, jump), 
               figsize=(6.5, 2.5)
               )

# %%
# save figure
fig.savefig("../FIGURES/illustration_MHE.pdf", bbox_inches='tight', pad_inches=0.01, dpi=1200)

# %%
plt.close("all")

# %%
