import numpy as np
import casadi as ca
import numpy.linalg as LA
from time import time
import contextlib

def build_arrivalcost(nx, options):
    if options["simple"]:
        return build_arrivalcost_simple(nx)
    if options["present"]:
        sq_params = ca.SX.sym("sqroot_sigma", nx*(nx+1)//2)
        Sigmainv_params = ca.SX.sym("sigmainv", nx*(nx+1)//2)
        sqroot = ca.SX.zeros(nx, nx)
        Sigmainv = ca.SX.zeros(nx, nx)
        idx = 0
        for i in range(nx):
            for j in range(i+1):
                sqroot[i,j] = sq_params[idx]
                Sigmainv[i,j] = Sigmainv_params[idx]
                Sigmainv[j,i] = Sigmainv_params[idx]
                idx += 1
        eta = Sigmainv_params
        # constraints = ca.vertcat(constraints, -constraints)
        if options["with_x0bar"]:
            x0bar = ca.SX.sym("x0bar", nx)
            eta = ca.vertcat(x0bar, eta)
        else:
            x0bar = ca.SX.zeros(nx)
        
        if options["with_square_root"]:
            eta = ca.vertcat(eta, sq_params)
            # make Equality constraints
            eye = ca.DM.eye(nx)
            Eq_constraints = ca.reshape(Sigmainv @ sqroot @ sqroot.T - eye, -1, 1)
            # make Inequality constraints
            Ineq_constraints = ca.SX(0)
        else:
            Eq_constraints = ca.SX(0)
            minors = []
            subindices = [[i,j] for i in range(nx) for j in range(i)] \
                         +   [[i] for i in range(nx)] # TODO maybe add mode
            for I in subindices:
                sub_mat = ca.hcat([ca.vcat([Sigmainv[i, j] for i in I]) for j in I])
                minor = ca.det(sub_mat)
                minors.append(minor)
            Ineq_constraints = -ca.vcat(minors)
            # print(Ineq_constraints)
        neta = eta.shape[0]
        eta_sym = ca.SX.sym("eta", neta)
        Sigmainv_sym = ca.Function("Sigmainv_fn", [eta], [Sigmainv])(eta_sym)
        x0bar_sym = ca.Function("x0bar_fn", [eta], [x0bar])(eta_sym)
        Eq_constraints_sym = ca.Function("Econstraints_fn", [eta], [Eq_constraints])(eta_sym)
        Ineq_constraints_sym = ca.Function("Iconstraints_fn", [eta], [Ineq_constraints])(eta_sym)
    else:
        eta_sym = ca.SX.sym("eta", 1, 1)
        Sigmainv_sym = ca.SX.zeros(nx, nx)
        x0bar_sym = ca.SX.zeros(nx)
        Eq_constraints_sym = ca.SX(0)
        Ineq_constraints_sym = ca.SX(0)
    return eta_sym, Sigmainv_sym, x0bar_sym, Ineq_constraints_sym, Eq_constraints_sym

def build_MHEconstraints(model, mest, mpred):
    dims = model["dims"]
    horizon = mest + mpred
    
    cost, variables = build_MHE(model, horizon)

    yest = variables["ys"][:mest, :]
    ypred = variables["ys"][mest:, :]

    xs_flat = ca.reshape(variables["xs"], -1, 1)
    ypred_flat = ca.reshape(ypred, -1, 1)
    optimization_variables = ca.vertcat(xs_flat, ypred_flat)
    
    optimality_conditions = ca.gradient(cost, optimization_variables)

    args = {
        "theta": variables["theta"],
        "Sigmainv": variables["Sigmainv"],
        "x0bar": variables["x0bar"],
        "u": variables["us"],
        "yest": yest,
        "ypred" : ypred,
        "x" : xs_flat,
    }

    constraints_fn = ca.Function("MHEoptCondition",
            list(args.values()), [optimality_conditions], list(args.keys()), ["val"])

    xdim = xs_flat.shape[0]
    return constraints_fn, xdim

def wnorm(x, W):
    return x.T @ W @ x

def build_MHE(model, horizon):
    dims = model["dims"]

    theta  = ca.SX.sym("theta", dims["theta"])
    Sigmainv  = ca.SX.sym("Sigmainv", dims["x"], dims["x"])
    x0bar  = ca.SX.sym("x0bar", dims["x"])

    Qinv = model["Qinv"](theta)
    Rinv = model["Rinv"](theta)

    
    x = ca.SX.sym(f"x_{0}", dims["x"])
    cost = wnorm(x-x0bar, Sigmainv)
    us, ys = [], []
    xs = [x]
    for i in range(horizon):
        xnext = ca.SX.sym(f"x_{i+1}", dims["x"])
        u = ca.SX.sym(f"u_{i}", dims["u"])
        y = ca.SX.sym(f"y_{i}", dims["y"])
        us.append(u)
        ys.append(y)
        xs.append(xnext)

        w = xnext - model["f"](x, u, theta, vect_type=ca.vcat)
        v = y - model["g"](xnext, theta)
        cost = cost + wnorm(w, Qinv) + wnorm(v, Rinv)
        x = xnext
        
    variables = {
        "theta": theta,
        "Sigmainv": Sigmainv,
        "x0bar": x0bar,
        "us": ca.hcat(us).T,
        "ys": ca.hcat(ys).T,
        "xs": ca.hcat(xs).T
    }
    return cost, variables

def build_eta(Sigmainv, x0, options):
    nx = len(x0)
    if not options["present"]:
        return np.array([0.])
    if options["simple"]:
        return build_eta_simple(Sigmainv, x0)
    sqroot = LA.inv(LA.cholesky(Sigmainv )) # Lower triangular matrix
    sqroot_params = np.zeros(nx*(nx+1)//2 )
    Sigmainv_params = np.zeros(nx*(nx+1)//2)
    idx = 0
    for i in range(nx):
        for j in range(i+1):
            sqroot_params[idx] = sqroot[i,j]
            Sigmainv_params[idx] = Sigmainv[i,j]
            idx += 1
    eta = Sigmainv_params
    if options["with_x0bar"]:
        eta = np.concatenate([x0, eta])
    if options["with_square_root"]:
        eta = np.concatenate([eta, sqroot_params])
    return eta

def MHE_wsensitivity(model, m, mpred, params_arrival_cost):
    nx = model["dims"]["x"]
    cost, variables = build_MHE(model, m)
    cost_fn = ca.Function("cost", list(variables.values()), [cost],
                                    list(variables.keys()), ["cost"])
    xs_tot = ca.SX.sym("xs", m+1, nx)
    ys_est = ca.SX.sym("ys", m-mpred, model["dims"]["y"])
    yhats = ca.SX.sym("ys", mpred, model["dims"]["y"])
    ys_tot = ca.vertcat(ys_est, yhats)
    us_tot = ca.SX.sym("us", m, model["dims"]["u"])

    theta_sym = ca.SX.sym("theta", model["dims"]["theta"])
    eta_sym, Sigmainv_sym, x0bar_sym, Ineq_constraints_eta, Eq_constraints_eta = \
        build_arrivalcost(nx, params_arrival_cost)
    alpha_sym = ca.vertcat(theta_sym, eta_sym)

    cost_sym = cost_fn(
        theta=theta_sym,
        Sigmainv=Sigmainv_sym,
        x0bar=x0bar_sym,
        us=us_tot,
        ys=ys_tot,
        xs=xs_tot
    )["cost"]

    yhat_flat = ca.reshape(yhats, -1, 1)
    variables_inner = ca.vertcat(
        ca.reshape(xs_tot, -1, 1),
        yhat_flat,
    )
    p = ca.vertcat(
        ca.reshape(us_tot, -1, 1),
        ca.reshape(ys_est, -1, 1),
        theta_sym,
        eta_sym
    )

    small_nlp = {
        "x": variables_inner,
        "f": cost_sym,
        "p": p
    }

    retrieve = ca.Function("retrieve", [variables_inner], [xs_tot, yhats])
    build_x = ca.Function("build_x", [xs_tot, yhats], [variables_inner])
    build_p = ca.Function("build_p", [us_tot, ys_est, alpha_sym], [p])

    # Compute residual for Gauss-Newton
    flatten_fn = ca.Function("r", [yhats], [yhat_flat])
    
    # Compute Linearization
    jac = ca.gradient(cost_sym, variables_inner)
    hess = ca.jacobian(jac, variables_inner)
    djac_dalpha = ca.jacobian(jac, alpha_sym)
    dy_dvar = ca.jacobian(yhat_flat, variables_inner)
    linearization_fn = ca.Function("linearization",
                    [variables_inner, p], [jac, hess, djac_dalpha, dy_dvar],
                    ["x", "p"], ["jac", "hess", "djac_dalpha", "dy_dvar"])

    # Constraints
    constraints_theta = ca.vcat(model["h"](theta_sym))
    Ineq_constraints = ca.vertcat(constraints_theta, Ineq_constraints_eta)
    Eq_constraints = Eq_constraints_eta
    h_fn = ca.Function("h_alpha", [alpha_sym], [Ineq_constraints, Eq_constraints])

    # Stack casadi functions
    functions = {
        "retrieve":retrieve,
        "build_x":build_x,
        "build_p":build_p,
        "linearization":linearization_fn,
        "flatten":flatten_fn,
        "h":h_fn
    }

    return small_nlp, functions

def compute_mhes(list_u, list_y, mest, alpha,
                 nlp_mhe, functions, warmstart=None, grad=True, exact=True, lbx=None, ubx=None):
    rt = {"lin":0.,  "opti":0., "deri":0., "final":0. }
    yhats, xs, residuals = [], [], []
    cost = 0.
    dy_dalphas = []
    N = len(list_u)
    t_begin = time()
    for i in range(N):
        utot = list_u[i]
        yest = list_y[i][:mest,:]
        ypred = list_y[i][mest:,:]
        if warmstart is None:
            x0 = None
            exact = True
        else:
            x0 = functions["build_x"](warmstart["x"][i], warmstart["yhat"][i])
        p = functions["build_p"](utot, yest, alpha)
        t0 = time()
        if exact:
            optimal_vars, status = quick_solve(nlp_mhe, p, x0, lbx=lbx, ubx=ubx)
            assert status == "Solve_Succeeded", "MHE failed to solve"
            x0 = optimal_vars
        t1 = time()
        if not exact or grad:
            lin = functions["linearization"](x=x0, p=p) #  ["jac", "inv_hess", "djac_dalpha", "dy_dvar"]
            inv_hess = LA.inv(lin["hess"].full())
            jac = lin["jac"].full()
            djac_dalpha = lin["djac_dalpha"].full()
            dy_dvar = lin["dy_dvar"].full()
        t2 = time()
        if not exact:
            optimal_vars = x0 - inv_hess @ jac
        t3 = time()
        if grad:
            dy_dalpha = - dy_dvar @ inv_hess @ djac_dalpha
            dy_dalphas.append(dy_dalpha)
        t4 = time()
        
        xtraj, yhat = functions["retrieve"](optimal_vars)
        x = xtraj.full()
        yhat = yhat.full()
        yhats.append(yhat)
        xs.append(x)
        
        error = yhat - ypred
        r = functions["flatten"](error).full()
        residuals.append(r)
        cost = cost + 0.5 * l2(r)

        t5 = time()
        
        rt["opti"] = rt["opti"] +  t1 - t0 + t3 - t2
        rt["lin"] = rt["lin"] + t2 - t1
        rt["deri"] = rt["deri"] + t4 - t3
        rt["final"] = rt["final"] + t5 - t4
    trajectories = {
        "yhat":yhats, 
        "x":xs,
        "r":residuals
    }
    rt["total"] = time() - t_begin
    return cost, trajectories, dy_dalphas, rt

def compute_cost(list_u, list_y, model, alpha, m, mpred, params_arrival_cost, 
                 Xmin=-np.inf, Xmax=np.inf):
    mest = m - mpred
    small_nlp, functions = MHE_wsensitivity(model, m, mpred, params_arrival_cost)
    lbx = np.ones(small_nlp["x"].shape) * Xmin
    ubx = np.ones(small_nlp["x"].shape) * Xmax
    nlp_mhe  = ca.nlpsol("S", "ipopt", small_nlp)

    warmstart = None # one could use it.. for now it does not seem needed or good.
    cost, _, _, _ = \
        compute_mhes(list_u, list_y, mest, alpha,
                 nlp_mhe, functions, warmstart=warmstart, grad=False, lbx=lbx, ubx=ubx)
    return cost

def l2(x):
    return np.sum(x**2)

def quick_solve(nlpsol, p, x0, lbx=None, ubx=None):
    params = dict()
    params["p"] = p
    if x0 is not None:
        params["x0"] = x0
    if lbx is not None:
        params["lbx"] = lbx
    if ubx is not None:
        params["ubx"] = ubx
    stdout = open('nul', 'w')
    with contextlib.redirect_stdout(stdout):
        sol = nlpsol(**params)
    status = nlpsol.stats()["return_status"]
    return sol["x"], status

def printmat(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print(f"{mat[i,j]:.2e} ", end=" ")
        print("")

def build_eta_simple(Sigmainv, x0):
    diagonal = np.array([Sigmainv[i,i] for i in range(len(x0))])
    eta = np.concatenate([x0, diagonal])
    return eta

def build_arrivalcost_simple(nx):
    diagonal = ca.SX.sym("sigmainv_diag", nx)
    Sigmainv_sym = ca.diag(diagonal)
    
    x0bar_sym = ca.SX.sym("x0bar", nx)
    eta_sym = ca.vertcat(x0bar_sym, diagonal)
    Eq_constraints_sym = ca.SX(0)

    diagmin = 1e-6
    xmax = 100 * np.ones(nx)
    xmin = -xmax
    Ineq_constraints_sym = ca.vertcat(
        diagmin-diagonal,
        xmin - x0bar_sym,
        x0bar_sym - xmax)

    return eta_sym, Sigmainv_sym, x0bar_sym, Ineq_constraints_sym, Eq_constraints_sym