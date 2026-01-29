# %%
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic('load_ext', 'autoreload')
ipython.run_line_magic('autoreload', '2')
ipython.run_line_magic('matplotlib', 'ipympl')
# %%
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle
import sys, os
from os.path import join, dirname
main_dir = dirname(os.getcwd())
sys.path.append(main_dir)
from UTILS import LorenzModel, simulation, prepare_data
# %%
dt = 0.02
sampling_time = 0.02
measure = [0,1]
jump = int(sampling_time / dt)
model = LorenzModel(dt, jump=jump, measure=measure, beta=2.)
dims = model["dims"]
list_theta_star = [
        np.array([10., 30.]),
        np.array([10., 20.]),
        np.array([20., 30.]),
        np.array([20., 20.]),
    ]
theta0 = np.array([15, 25]) # theta_star = [10., 30., 2.]
ntraj = 50

# %%
# Define the method
from SOURCE import whole_Method

simple_arrival_cost = True
# For tayloreed method
wipopt = False
with_square_root = True
niter = 20
# %%
m = 10
rng = np.random.default_rng(1234)
nsamples = 10

# %%
# Define how to generate data
def generate_data(rng, model, theta, dt, m,
                  T_per_traj=3.5,
                  ntraj = 100,
                  sample_every=1.
                  ):
    dims = model["dims"]
    N_per_traj = int(T_per_traj / dt)
    U, Y = [], []
    for i in range(ntraj):
        x0 = rng.uniform(-10, 10, dims["x"])
        us = np.zeros((N_per_traj, dims["u"]))
        ws = rng.uniform(-0.25, 0.25, (N_per_traj, dims["x"]))
        vs = rng.uniform(-1, 1, (N_per_traj, dims["y"]))
        ys = simulation(model, theta, us, ws, vs, x0)
        U.append(us)
        Y.append(ys)
    step_discrete = int(sample_every / dt)
    list_u, list_y = prepare_data(U, Y, m, step=step_discrete)
    return list_u, list_y

# %%
Big_List = []
for theta_star in list_theta_star:
    print(f"Starting theta_star = {theta_star}")
    list_saving = []
    for i in range(nsamples):
        print(f" {i+1}/{nsamples} -th realization")
        # print("Generating data...")
        list_u, list_y = generate_data(rng, model, theta_star, dt, m,
                                        ntraj = ntraj,
                                        sample_every=0.5,
                                        )
        # print("Build estimate...")
        t0 = time()
        alpha, cost, status = whole_Method(
                    list_u, list_y, model, theta0,
                    with_square_root = with_square_root,
                    simple_arrival_cost = simple_arrival_cost,
                    niter = niter,
                    wipopt = wipopt,
                    smart_eta0=False,
                    recompute_cost=True,
                    verbose = 1
                    )
        runtime = time() - t0
        theta = alpha[:len(theta0)]
        eta = alpha[len(theta0):]
        saving = {
            "theta": theta,
            # "eta": eta,
            "runtime": runtime,
            # "status": status,
        }
        list_saving.append(saving)
        for key, value in saving.items():
            print(f"       {key} = {value}")
    Big_List.append((theta_star, list_saving))
# %%
# Saving the objects:
if not os.path.exists("data_to_plot"):
    os.makedirs("data_to_plot")
with open("data_to_plot/lorenz.pkl", 'wb') as f:
    pickle.dump(Big_List, f)
# %%
