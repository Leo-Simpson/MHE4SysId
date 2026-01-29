import numpy as np
import casadi as ca
import contextlib

from .misc_method import build_MHEconstraints, build_arrivalcost
from .nlp import NLP


def ipopt_Method(list_u, list_y, model, warmstart, mpred, params_arrival_cost, regularize=0., Xmin=-np.inf, Xmax=np.inf):
    nlp, retrieve, retrieve_x = Build_Method(
        list_u, list_y, model, warmstart, mpred, params_arrival_cost, regularize=regularize,
        Xmin=Xmin, Xmax=Xmax)
    nlp.presolve()
    # store ipopt outputs in a text file, in a dedicated folder
    if not os.path.exists('ipopt_output'):
        os.makedirs('ipopt_output')
    stdout = open('ipopt_output/full_ipopt', 'w')
    with contextlib.redirect_stdout(stdout):
        status, alpha = solve_problem(
            nlp, retrieve, retrieve_x, model["g"])

    return alpha, status

def Build_Method(list_u, list_y, model, warm_start, mpred, params_arrival_cost, regularize=0., Xmin=-np.inf, Xmax=np.inf):
    N = len(list_u)
    m = len(list_u[0])
    mest = m - mpred
    dims = model["dims"]

    theta0 = warm_start["alpha"][:dims["theta"]]
    eta0 = warm_start["alpha"][dims["theta"]:]

    constr_fn, xdim = build_MHEconstraints(model, mest, mpred)
    nlp = NLP()
    theta = ca.SX.sym("theta", dims["theta"])
    nlp.add_var(theta, theta0)
    
    eta, Sigmainv, x0bar, Heta, Geta = build_arrivalcost(dims["x"], params_arrival_cost)
    nlp.add_var(eta, eta0)
    nlp.add_eqconstr(Geta)
    nlp.add_ineqconstr(Heta)

    
    dtheta = theta - theta0
    deta = eta - eta0
    nlp.add_cost( 
        regularize * (ca.dot(dtheta, dtheta) + ca.dot(deta, deta))
    )

    htheta = model["h"](theta)
    nlp.add_ineqconstr(ca.vcat(htheta) )
    xtrajs = []
    for i in range(N):
        
        yest= list_y[i][:mest]
        ypred = list_y[i][mest:]
        utot = list_u[i]

        yhat = ca.SX.sym(f"yhat{i}", mpred, dims["y"])
        yhat0 = warm_start["yhat"][i]
        nlp.add_var(yhat, yhat0, lb=Xmin, ub=Xmax)
        error = ypred - yhat
        nlp.add_cost(0.5 * ca.sumsqr(error))

        xtraj = ca.SX.sym("xs", m+1, dims["x"]) # additional variables for the constraints
        xtrajs.append(xtraj)
        xtraj0 = warm_start["x"][i]
        
        x_flat = ca.reshape(xtraj, -1, 1)
        x0_flat = ca.reshape(xtraj0, -1, 1)
        
        nlp.add_var(x_flat, x0_flat, lb=Xmin, ub=Xmax)
        constr = constr_fn(
            theta = theta,
            Sigmainv = Sigmainv,
            x0bar = x0bar,
            u = utot,
            yest = yest,
            ypred = yhat,
            x = x_flat
            )["val"]
        nlp.add_eqconstr(constr)

    retrieve = ca.Function("retrieve", [nlp.variables], [theta, eta, Sigmainv, x0bar], ["nlpx"], ["theta", "eta", "Sigmainv", "x0bar"])
    retrieve_x = ca.Function("retrieve", [nlp.variables], xtrajs)
    return nlp, retrieve, retrieve_x

def solve_problem(nlp, retrieve, retrieve_x, g):
    # nlp.presolve()
    sol = nlp.solve()
    cost = sol["f"].full().squeeze()
    ret = retrieve(nlpx=sol["x"])
    status = nlp.nlpsol.stats()["return_status"]

    theta = ret["theta"].full()[:, 0]
    eta = ret["eta"].full()[:, 0]
    alpha = np.concatenate([theta, eta])

    return status, alpha



