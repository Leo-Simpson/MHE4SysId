import numpy as np
import casadi as ca

from .misc_method import build_eta, MHE_wsensitivity, compute_mhes
from .ipopt import ipopt_Method

def vanilla_IPOPT(list_u, list_y, model, theta0, mpred=1, verbose=1,
                 arrival_cost=True, with_square_root=False,
                 simple_arrival_cost=False
                 ):
    m = len(list_u[0])
    params_arrival_cost = {
        "present": arrival_cost,
        "with_square_root": with_square_root,
        "simple": simple_arrival_cost,
        "with_x0bar": True}
    
    if verbose: print("  Build warm start...")

    Sigmainv0, x00 = np.eye(model["dims"]["x"]), np.zeros(model["dims"]["x"])
    eta0 = build_eta(Sigmainv0, x00, params_arrival_cost) 
    alpha0 = np.concatenate([theta0, eta0])
    mest = m - mpred
    small_nlp, functions = MHE_wsensitivity(model, m, mpred, params_arrival_cost)
    nlp_mhe = ca.nlpsol("S", "ipopt", small_nlp)
    warm_start1 = {
        "alpha": alpha0,
        "x": [np.zeros((m+1, model["dims"]["x"]))]*len(list_u),
        "yhat": [np.zeros((mpred, model["dims"]["y"]))]*len(list_u)
    }
    mest = m - mpred
    _, trajectories, _, _ = \
        compute_mhes(list_u, list_y, mest, alpha0,
                    nlp_mhe, functions, grad=False,
                    warmstart=warm_start1)
    warm_start2 = {
        "alpha":alpha0,
        "x":trajectories["x"],
        "yhat":trajectories["yhat"]
    }
    if verbose: print("  IPOPT method...")
    alpha, status = \
        ipopt_Method(list_u, list_y, model, warm_start2, mpred, params_arrival_cost)
    if verbose: print(f"  IPOPT status : {status}")

    return alpha, status

