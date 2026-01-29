# %%
import sys, os
from os.path import join, dirname
main_dir = dirname(os.getcwd())
sys.path.append(main_dir)

import numpy as np
from time import time
import matplotlib.pyplot as plt
import casadi as ca
import pickle

from UTILS import latexify
latexify()

colors = [
    "blue", "green", "orange", "purple"
]
color_legend = "grey"
markersize = 4
# %%
file_name = "data_to_plot/lorenz.pkl"
assert os.path.exists(file_name), f"Run first the script 'estimates_Lorenz(generate)' to generate data."
with open(file_name, "rb") as f:
    Big_List = pickle.load(f)


# %%
theta0 = np.array([15, 25])
theta_1_min = 10
theta_1_max = 20
theta_2_min = 20
theta_2_max = 30
# %%

fig, axs = plt.subplots(1, 2,
                    figsize=(4.2, 2.5),
                    gridspec_kw=
                    {'width_ratios': [3, 1]}
                        )
ax = axs[0]
ax_legend = axs[1]

ax.grid(True)
ax.set_xlabel(r"$\theta_1$")
pad = 6
ax.set_ylabel(r"$\theta_2$")
ax.set_xlim([theta_1_min - pad, theta_1_max + pad])
ax.set_ylim([theta_2_min - pad, theta_2_max + pad])
ax.plot(theta0[0], theta0[1], "o", markersize=4, color="black",
        label=r"Initial guess $\theta^0$") 
for j, (theta_star, list_savings) in enumerate(Big_List):
    print(theta_star)
    ax.plot(theta_star[0], theta_star[1], "*", markersize=15,
            alpha=0.5, color=colors[j])
    for saving in list_savings:
        theta = saving["theta"]
        ax.plot(theta[0], theta[1], marker="x", markersize=markersize, color=colors[j])

# Outside of the plot, only for the legend
ax.plot(0., 0., "x", marker="x", markersize=markersize, label=r"Estimates $\hat{\theta}$", color=color_legend)
ax.plot(0., 0., "*", markersize=15,
        label= r"Ground truths $\theta^\star$",
        color=color_legend, alpha=0.5)
h, l = ax.get_legend_handles_labels()
ax_legend.axis("off")
ax_legend.legend(h, l, loc="center left", ncol=1,
                 bbox_to_anchor=(-0.4, 0.5)
                 )

# %%
# save Figure
fig.savefig("../FIGURES/estimates_Lorenz", bbox_inches='tight', pad_inches=0.01)

# %%
plt.close("all")

# %%