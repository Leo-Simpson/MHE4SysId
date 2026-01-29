import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib
def latexify():
    params = {
        'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': True,
        'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrowPatch(
        (0, 0.5*height), (width, 0.5*height),
                    mutation_scale=10,
                    arrowstyle='<->')
    return p

handler_map={mpatches.FancyArrowPatch : HandlerPatch(patch_func=make_legend_arrow)}


def mhe_plot(
    subT, subY, Tcont, Ycont,
    xticks=None,
    xlims=None,
    mhe=None,
    figsize=(4.8, 2.8)
):
    T_conc = np.concatenate(subT, axis=0)
    Y_conc = np.concatenate(subY, axis=0)

    color_est = "yellow"
    color_mhe = "orange"
    color_measurements = "green"
    color_error = "red"
    color_pred = "purple"
    color_arrival_cost = "pink"
    markersize = 8.

    fig, axs = plt.subplots(2,1, figsize=figsize,
        gridspec_kw={'height_ratios':[14, 1]}
    )
    ax = axs[0]

    # ax.set_title("Illustration of the method")
    ax.set_xlabel("Time (s)")
    # ax.set_ylabel(r"State $x_2$")
    if xlims is not None:
        ax.set_xlim(xlims)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.plot(Tcont, Ycont, "-", label="System",  alpha=0.2, color="blue")
    ax.plot(T_conc, Y_conc, ".", markersize=markersize, label="Measurements",  color=color_measurements)
    wlabel = True
    for seqT in subT:
        step = seqT[1] - seqT[0]
        t0 = seqT[0]
        t1 = seqT[-2] + step/4
        t2 = seqT[-1] + step
        label1, label2 = None, None
        if wlabel:
            label1 = "Estimation window"
            label2 = "Prediction window"
        ax.axvspan(t0, t1, alpha=0.2, color=color_est, label=label1)
        ax.axvspan(t1, t2, alpha=0.2, color=color_pred, label=label2)
        wlabel = False

    if mhe is not None:
        subhaty, subhaty_t, last_points, y0bar, jump = mhe
        last_ests = [(seqhaty_t[-jump], seqhaty[-jump]) for (seqhaty, seqhaty_t) in zip(subhaty, subhaty_t)]
        wlabel = True
        for meas, est in zip(last_points, last_ests):
            label = None
            if wlabel:
                label = "Prediction error"
            # ax.plot([meas[0], est[0]], [meas[1], est[1]], "-",
            #         color=color_error, alpha=0.7,
            #         label=label)
            # ax.annotate("",
            #             xy=meas, xytext=est,
            #             arrowprops=dict(
            #                 arrowstyle="<->",
            #                 color=color_error),
            #             label=label)
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    (meas[0], meas[1][0]), (est[0], est[1][0]),
                    mutation_scale=10,
                    arrowstyle='<->',
                    color=color_error,
                    label=label ) )
            wlabel = False
        
        ax.axhline(y=y0bar, color=color_arrival_cost, linestyle="--", alpha=0.5)
        wlabel = True
        for seqhaty, seqhaty_t in zip(subhaty, subhaty_t):
            label1, label2, label3 = None, None, None
            if wlabel:
                label1 = "Trajectory estimation"
                label2 = "Prediction"
                label3 = r" $\bar{x}_0$ (in the arrival cost)"
            wlabel = False
            t0bar = seqhaty_t[0]
            ax.plot(t0bar , y0bar, "o",
                    label=label3, color=color_arrival_cost, alpha=0.6, markersize=10.)
            ax.plot(seqhaty_t, seqhaty, "-.", label=label1, color=color_mhe)
            index = len(seqhaty_t) - jump
            last_est = (seqhaty_t[index], seqhaty[index])
            ax.plot(last_est[0], last_est[1], ".", label=label2,
                    color=color_mhe, markersize=markersize)

    ax4legend = axs[1]
    ax4legend.axis("off")
    h, l = ax.get_legend_handles_labels()
    ax4legend.legend(h, l, loc="center left", ncol=3,
                     handler_map=handler_map,
                     bbox_to_anchor=(-0.1, -4.8)
                     )


    return fig

