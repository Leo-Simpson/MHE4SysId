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
from UTILS import LorenzModel, simulation, parse, subsequences2, latexify

# %%
rng = np.random.default_rng(1234)

# %%
dt = 0.02
sampling_time = 0.02
measure = [0,1]
jump = int(sampling_time / dt)
model_cont = LorenzModel(dt, measure=measure)
model_cont_full = LorenzModel(dt, measure=[0,1,2])
model = LorenzModel(dt, jump=jump, measure=measure)
dims = model["dims"]
theta_star = np.array([10., 25., 2.]) # [sigma, rho, beta]

# %%
# Generate data
ntraj = 10
T_per_traj = 3.5
T_remove = 2
N_remove = int(T_remove / dt)
N_per_traj = int(T_per_traj / dt)+N_remove
Ucont = np.zeros((N_per_traj, dims["u"]))
vs_full = np.zeros((N_per_traj, dims["x"]))
Tcont = np.arange(N_per_traj-N_remove) * dt
list_Ycont = []
list_Y, list_T = [], []
for i in range(ntraj):
    x0 = rng.uniform(-10, 10, dims["x"])
    ws = rng.uniform(-0.25, 0.25, (N_per_traj, dims["x"]))
    vs = rng.uniform(-1, 1, (N_per_traj, dims["y"]))
    ys_cont  = simulation(model_cont_full, theta_star, Ucont, ws, vs_full, x0)
    ys = simulation(model_cont, theta_star, Ucont, ws, vs, x0)
    ys = ys[N_remove:]
    ys_cont = ys_cont[N_remove:]
    list_Ycont.append(ys_cont)
    list_Y.append(parse(ys, jump))
    list_T.append(parse(Tcont, jump))

# %%
# Process data
m = 10
sample_every = 0.5
step = int(sample_every / dt)
list_subT, list_subY = subsequences2(list_T, list_Y, m, step)

# %%
# Plot data
markersize = 3
latexify()
fig, axs = plt.subplots(dims["x"]+1, figsize=(4.7, 4.2),
    # sharex=True,
    gridspec_kw={'height_ratios':[2 for i in range(dims["x"])]+[1]}
)
i = 2 # rng.integers(0, ntraj)
Ycont = list_Ycont[i]
subT = list_subT[i]
subY = list_subY[i]
T_conc = np.concatenate(subT, axis=0)
Y_conc = np.concatenate(subY, axis=0)
for j in range(3):
    ax = axs[j]
    # ax.set_title("Illustration of MHE on Lorenz attractor")
    if j < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$x_" +str(j+1)+ r"$")

    ax.plot(Tcont, Ycont[:,j], "-", label="States",  alpha=0.2, color="blue")

    wlabel = True
    for seqT, seqY in zip(subT, subY):
        step = seqT[1] - seqT[0]
        t0 = seqT[0]
        t1 = seqT[-2] + step/4
        t2 = seqT[-1] + step
        label1, label2 = None, None
        if wlabel:
            label1 = "Estimation window"
            label2 = "Prediction window"
        ax.axvspan(t0, t1, alpha=0.2, color="orange", label=label1)
        ax.axvspan(t1, t2, alpha=0.2, color="purple", label=label2)

        wlabel = False
    if j in measure:
        ax.plot(T_conc, Y_conc[:, j], ".", markersize=markersize, label="Measurements",  color="green")

ax4legend = axs[-1]
ax4legend.axis("off")
h, l = axs[1].get_legend_handles_labels()
ax4legend.legend(h, l, loc="center left", ncol=2,
                 bbox_to_anchor=(-0.1, -0.6))


# %%
fig.savefig("../FIGURES/trajectories_Lorenz.pdf", bbox_inches='tight', pad_inches=0.02)

# %%
plt.close("all")

# %%