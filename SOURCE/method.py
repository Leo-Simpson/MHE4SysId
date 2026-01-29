import numpy as np

from .initial_guess import create_initial_guess
from .misc_method import build_eta, compute_cost
from .opti import iterate
from .ipopt import ipopt_Method
from .lti_method import LTI_Method

def whole_Method(list_u, list_y, model, theta0, mpred=1, niter=20, verbose=1, recompute_cost=True,
                 smart_eta0=True, arrival_cost=True, with_square_root=False,
                 lti=False, wipopt=False,
                 simple_arrival_cost=False,
                 Xmin=-np.inf, Xmax=np.inf
                 ):
    m = len(list_u[0])
    
    params_arrival_cost = {
        "present": arrival_cost,
        "with_square_root": with_square_root,
        "simple": simple_arrival_cost}
    shift_data = False # (lti and arrival_cost and mpred==1)
    if shift_data:
        params_arrival_cost["with_x0bar"] = False
        mean_u = np.mean(list_u, axis=0)
        mean_y = np.mean(list_y, axis=0)
        list_u_ = [u - mean_u for u in list_u]
        list_y_ = [y - mean_y for y in list_y]
    else:
        params_arrival_cost["with_x0bar"] = True
        list_u_ = list_u
        list_y_ = list_y
    
    # Prepare warm start
    if smart_eta0:
        if verbose: print("  Preparing warm start...")
        Sigmainv0, x00 = create_initial_guess(list_u_, list_y_, model, theta0, m, mpred, Xmin=Xmin, Xmax=Xmax)
    else:
        Sigmainv0, x00 = np.eye(model["dims"]["x"]), np.zeros(model["dims"]["x"])

    eta0 = build_eta(Sigmainv0, x00, params_arrival_cost) 
    alpha0 = np.concatenate([theta0, eta0])
    if lti:
        if verbose: print(" Subspace Method (for LTI only)...")
        alpha, status, cost = LTI_Method(list_u_, list_y_, model, alpha0, params_arrival_cost,
                                         mpred=mpred, verbose=verbose)
        if verbose: print(f"  Subspace cost = {cost}, status : {status},")
    else:
        if verbose: print("  Self-made opti...")
        solution_ws2 = iterate(niter, list_u_, list_y_, model, alpha0, m, mpred, params_arrival_cost,
                            verbose=verbose,  Xmin=Xmin, Xmax=Xmax)
        status = "not available"
        if wipopt:
            if verbose: print("  Final IPOPT method...")
            alpha, status = \
                ipopt_Method(list_u_, list_y_, model, solution_ws2, mpred, params_arrival_cost, Xmin=Xmin, Xmax=Xmax)
            if verbose: print(f"  Final IPOPT status : {status}")
        else:
            alpha = solution_ws2["alpha"]

    if recompute_cost: cost = compute_cost(
                                list_u_, list_y_, model, alpha, m, mpred, params_arrival_cost,
                                Xmin=Xmin, Xmax=Xmax)
    if verbose: print(f"  Final cost = {cost:.4e}")
    return alpha, cost, status

