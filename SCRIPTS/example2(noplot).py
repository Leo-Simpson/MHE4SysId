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

from UTILS import simpleModel, simulation, prepare_data, resample

# %%
rng = np.random.default_rng(1234)

# %%
model = simpleModel(n=2)
dims = model["dims"]
theta_star = np.array([0.9, 0.9]) # [a, b]
theta0 = np.array([0.5, 0.5]) # [a, b]


# %%
# Generate data
# Choose some input sequence (always the same)
nsteps = 5
len_steps = 50
us = np.concatenate([
        (-1)**(i)*np.ones((len_steps, dims["u"])) for i in range(nsteps)
        ])

ntraj = 100
N_per_traj = nsteps * len_steps
noise_scale = 1.*1e-3
U, Y = [], []
for i in range(ntraj):
    x0 = 0.* np.ones(dims["x"]) + 5. * rng.normal(0, 1, dims["x"])
    ws = rng.normal(0, 1, (N_per_traj, dims["x"])) * noise_scale
    vs = rng.normal(0, 1, (N_per_traj, dims["y"])) * noise_scale
    ys = simulation(model, theta_star, us, ws, vs, x0)
    Y.append(ys)
    U.append(us)
# %% 
# Prepare data
mest, mpred = 5, 1

m = mest + mpred
Nseq = 500
list_u, list_y = prepare_data(U, Y, m)
list_u, list_y = resample(
    list_u, list_y, rng, Nseq)
print(f"Number of sequences N = {Nseq}")
print(f"Horizon m = {m}")


# %%
from SOURCE import whole_Method

alpha, cost, status = whole_Method(
            list_u, list_y, model, theta0,
            wipopt=False, with_square_root=False,
            lti=True,
            verbose=2)

# %%
theta = alpha[:len(theta0)]
print(f"theta = {theta}")

# %%
from SOURCE import whole_Method

alpha, cost, status = whole_Method(
            list_u, list_y, model, theta0,
            wipopt=False, with_square_root=False,
            # lti=True,
            verbose=2)

# %%
theta = alpha[:len(theta0)]
print(f"theta = {theta}")
# %%
