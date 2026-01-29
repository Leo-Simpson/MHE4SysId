import os
import numpy as np
import numpy.linalg as LA
import casadi as ca
import contextlib
from .misc_method import build_MHE

def create_MHE_solver(model, theta0, m, mpred):
    mest = m - mpred
    nx = model["dims"]["x"]
    cost, variables = build_MHE(model, m)
    cost_fn = ca.Function("cost", list(variables.values()), [cost],
                                    list(variables.keys()), ["cost"])
    xs_tot = ca.SX.sym("xs", m+1, nx)
    ys_est = ca.SX.sym("ys", m-mpred, model["dims"]["y"])
    ys_pred = ca.SX.sym("ys", mpred, model["dims"]["y"])
    ys_tot = ca.vertcat(ys_est, ys_pred)
    us_tot = ca.SX.sym("us", m, model["dims"]["u"])
    cost_sym = cost_fn(
        theta=theta0,
        Sigmainv=np.zeros((nx, nx)),
        x0bar=np.zeros(nx),
        us=us_tot,
        ys=ys_tot,
        xs=xs_tot
    )["cost"]

    variables_inner = ca.vertcat(
        ca.reshape(xs_tot, -1, 1),
        ca.reshape(ys_pred, -1, 1),
    )
    p = ca.vertcat(
        ca.reshape(us_tot, -1, 1),
        ca.reshape(ys_est, -1, 1),
    )

    small_nlp = {
        "x": variables_inner,
        "f": cost_sym,
        "p": p
    }

    retrieve = ca.Function("retrieve", [variables_inner], [xs_tot, ys_pred])
    build_p = ca.Function("build_p", [us_tot, ys_est], [p])

    return small_nlp, retrieve, build_p

def quick_solve(nlpsol, p, lbx, ubx):
    # store ipopt outputs in a text file, in a dedicated folder
    if not os.path.exists('ipopt_output'):
        os.makedirs('ipopt_output')
    stdout = open('ipopt_output/quicksol.txt', 'w')
    with contextlib.redirect_stdout(stdout):
        sol = nlpsol(p = p, lbx=lbx, ubx=ubx )
    status = nlpsol.stats()["return_status"]
    assert status == "Solve_Succeeded"
    return sol["x"]

def create_initial_guess(list_u, list_y, model, theta0, m, mpred, Xmin=-np.inf, Xmax=np.inf):
    mest = m - mpred

    small_nlp, retrieve, build_p = create_MHE_solver(model, theta0, m, mpred)
    nlpsol = ca.nlpsol("S", "ipopt", small_nlp)

    lbx = np.ones(small_nlp["x"].shape) * Xmin
    ubx = np.ones(small_nlp["x"].shape) * Xmax

    # guess on trajectories
    x0s = []
    N = len(list_u)
    for i in range(N):
        utot = list_u[i]
        yest = list_y[i][:mest]
        p = build_p(utot, yest)
        solution = quick_solve(nlpsol, p, lbx, ubx)
        xtraj, yhat = retrieve(solution)
        x0s.append(xtraj.full()[0,:])
    # guess on arrival cost
    Sigma, x0bar = mycov(x0s)
    Sigmainv = LA.inv(Sigma)
    return Sigmainv, x0bar

def mycov(xs):
    m = np.mean(xs, axis=0)
    cov = np.mean([np.outer(x-m, x-m) for x in xs], axis=0)
    return cov, m