import numpy as np
import casadi as ca
import contextlib
from .nlp import NLP


def simple_MHE(fmodel, gmodel, N, ks, ys, x0bar, Sigmainv, Qinv, Rinv):
    """
    Simple Moving Horizon Estimation (MHE) algorithm.
    In this special case, the measurements are accessible only at times ks
    but inputs are available at all times
    """
    nx = Qinv.shape[0]
    warm_start = np.zeros((N, nx))

    xs = ca.SX.sym("x", (N, nx))
    x = xs[0, :].T
    cost = wnorm(x - x0bar, Sigmainv)
    for i in range(N-1):
        x = xs[i, :].T
        xnext = xs[i+1, :].T
        w = xnext - fmodel(x)
        cost = cost + wnorm(w, Qinv)
        if i in ks:
            idx = np.where(ks == i)[0][0]
            v = ys[idx] - gmodel(x)
            cost = cost + wnorm(v, Rinv)
        
    nlp = NLP()
    nlp.add_var(xs, warm_start)
    nlp.add_cost(cost)
    nlp.presolve()
    stdout = open('nul', 'w')
    with contextlib.redirect_stdout(stdout):
        sol = nlp.solve()

    ys_hat = ca.vcat([gmodel(xs[i]).T for i in range(N)])
    ys_sol = ca.Function("retrieve", [nlp.variables], [ys_hat])(sol["x"]).full()
    xs_sol = ca.Function("retrievex", [nlp.variables], [xs])(sol["x"]).full()
    
    return ys_sol, xs_sol

def wnorm(x, W):
    return (x.T @ W @ x)[0,0]