
import numpy as np
import numpy.linalg as LA
import casadi as ca
from time import time
import contextlib
from .misc_method import MHE_wsensitivity, compute_mhes, printmat

def iterate(niter, list_u, list_y, model, alpha0, m, mpred, params_arrival_cost, verbose=1, Xmin=-np.inf, Xmax=np.inf):
    rtimes = {}
    t0 = time()
    pen_increase = 0.1
    pen_min = 1e-2
    exactMHE = True
    mest = m - mpred
    small_nlp, functions = MHE_wsensitivity(model, m, mpred, params_arrival_cost)
    lbx = np.ones(small_nlp["x"].shape) * Xmin
    ubx = np.ones(small_nlp["x"].shape) * Xmax
    nlp_mhe = ca.nlpsol("S", "ipopt", small_nlp)

    solution = {}
    solution["alpha"] = alpha0
    solution["cost"] = np.inf
    solution["x"] = [np.zeros((m+1, model["dims"]["x"]))]*len(list_u)
    solution["yhat"] = [np.zeros((mpred, model["dims"]["y"]))]*len(list_u)
    pen = pen_min
    if verbose>1: print(f"  initialization        cost = {solution['cost']:.2e}")
    for iter in range(niter):
        solution = step(list_u, list_y, solution, mest,
            nlp_mhe,  functions, rtimes, pen=pen, exactMHE=exactMHE, lbx=lbx, ubx=ubx)
        if solution["accepted"]:
            pen = pen_min
        else:
            pen = pen + pen_increase
        acceptance = "accepted" if solution["accepted"] else "rejected"
        if verbose>1: print(f"      iteration {iter+1:02}/{niter}    cost = {solution['cost']:.4e}  ({acceptance})")    
        if verbose>2:
            theta = solution['alpha'][:model["dims"]["theta"]]
            eta = solution['alpha'][model["dims"]["theta"]:]
            print(f"     theta= {theta}")
            print(f"     eta= {eta}")
    if verbose: print(f"      final cost = {solution['cost']:.4e}")
    tf = time()
    rtimes["total"] = tf - t0
    if verbose>2:
        print("    Run times :")
        for operation, rt in rtimes.items():
            print(f"      {operation} : {rt}")
    return solution

def step(list_u, list_y, solution0, mest,
         nlp_mhe,  functions, rtimes, pen=0., exactMHE=True, lbx=None, ubx=None):
    alpha0 = solution0["alpha"]
    warm_start = {"x":solution0["x"], "yhat":solution0["yhat"]}
    if "dyhat_dalpha" not in solution0:
         # That should be only for the first iteration
        solution0 = new_solution(list_u, list_y, mest, nlp_mhe, functions, alpha0, rtimes,
                                     warmstart=warm_start, exact=True, lbx=lbx, ubx=ubx)
    # Perform Gauss-Newton step
    gradient, hessian = \
            gn_qp(solution0["r"], solution0["dyhat_dalpha"])
    hessian = treat_hessian(hessian, pen=pen)
    alpha, status = solve_qp(gradient, hessian, alpha0, functions["h"])
    assert status == "Solve_Succeeded", "QP failed to solve"
    candidate_solution = new_solution(list_u, list_y, mest, nlp_mhe, functions, alpha, rtimes,
                                          warmstart=warm_start, exact=exactMHE, lbx=lbx, ubx=ubx)
    acceptance = candidate_solution["cost"] <= solution0["cost"] + 1e-6
    if acceptance:
        solution = candidate_solution
    else:
        solution0 = new_solution(list_u, list_y, mest, nlp_mhe, functions, alpha0, rtimes,
                                     warmstart=warm_start, exact=exactMHE, lbx=lbx, ubx=ubx)
        # this is needed only in the case of inexact MHE
        solution = solution0
    solution["accepted"] = acceptance
    return solution

def new_solution(list_u, list_y,  mest, nlp_mhe, functions, alpha, rt,
                 warmstart=None, exact=True, lbx=None, ubx=None):
    cost, trajectories, dyhats, rtMHE = \
        compute_mhes(list_u, list_y, mest, alpha,
                    nlp_mhe, functions, grad=True,
                    warmstart=warmstart,
                    exact=exact, lbx=lbx, ubx=ubx)
    solution = {
        "alpha":alpha,
        "x":trajectories["x"],
        "yhat":trajectories["yhat"],
        "r":trajectories["r"],
        "dyhat_dalpha":dyhats,
        "cost":cost, # remark: this was actually the cost for the previous iteration in terms of alpha
    }

    for operation, t in rtMHE.items():
        name = "MHE"+operation
        if not name in rt:
            rt[name] = 0.
        rt[name] = rt[name] + t

    return solution

def gn_qp(rs, drs):
    N = len(rs)
    Gradient, gn_Hessian = 0., 0.

    for i in range(N):
        r = rs[i]
        dr = drs[i]
        
        grad = dr.T @ r
        gn_hessian = dr.T @ dr

        Gradient = Gradient + grad
        gn_Hessian = gn_Hessian + gn_hessian
    
    return Gradient, gn_Hessian

def solve_qp(g, H, x_bar, h_fn):
    """
        Solve the QP as follows:
            min_dx 0.5 * dx^\top H dx + g^\top dx + pen * (dx^\top dx)
            s.t. 
                h(x_bar + dx) <= 0.
    """
    
    n = len(x_bar)
    x = ca.SX.sym("x", n)
    dx = x - x_bar
    f = 0.5 * ca.mtimes([dx.T, H, dx]) + ca.mtimes([g.T, dx])
    ineq, eq = h_fn(x)
    constraints = ca.vertcat(ineq, eq)
    nineq, neq = ineq.shape[0], eq.shape[0]
    lbg = np.concatenate([ -np.inf*np.ones(nineq)  , np.zeros(neq)])
    ubg = np.concatenate([ np.zeros(nineq)  , np.zeros(neq)])
    
    nlp = { "x":x, "f":f, "g":constraints}
    solver = ca.nlpsol("solver", "ipopt", nlp)
    stdout = open('nul', 'w')
    with contextlib.redirect_stdout(stdout):
        sol = solver(
            x0=x_bar,
            lbg = lbg, ubg = ubg
            )
    status = solver.stats()["return_status"]
    x = sol["x"].full().squeeze()
    return x, status

def treat_hessian(hessian, pen=0.):
    # print("Hessian =")
    # printmat(hessian)
    # eigenvalues, eigenvectors = LA.eigh(hessian)
    # print(f"eigenvalues = {eigenvalues}")
    hessian = hessian + pen * np.eye(hessian.shape[0])
    return hessian
