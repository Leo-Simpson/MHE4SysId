import numpy as np
import casadi as ca
import numpy.linalg as LA
import contextlib
import os

from .misc_method import build_arrivalcost, build_MHE, printmat
from .nlp import NLP

def LTI_Method(list_u, list_y, model, alpha0, params_arrival_cost, mpred=1, verbose=0):
    m = len(list_y[0])
    mest = m - mpred
    dims = model["dims"]
    psi_fn = build_psi_fn(model, m, mpred)
    
    Hessian, gradient, offset =  condense(list_u, list_y, mpred=mpred)
    alpha, status, cost = efficientMethod(Hessian, gradient, psi_fn, dims, model["h"], alpha0, params_arrival_cost, m, mpred=mpred, offset=offset)
    cost = len(list_y) * cost # len(list_y) * cost
    return alpha, status, cost

def efficientMethod(H, g, psi_fn, dims, h_theta, alpha0, params_arrival_cost, m, mpred=1, offset=0.):
    scaling =  1e4
    
    mest = m - mpred
    nlp = NLP()
    
    # build symbolic variables
    theta_sym = ca.SX.sym("theta", dims["theta"])
    eta_sym, Sigmainv, x0bar, Heta, Geta  = build_arrivalcost(dims["x"], params_arrival_cost)
    nlp.add_ineqconstr(Heta)
    nlp.add_eqconstr(Geta)

    # build constraints
    htheta_sym = ca.vcat(h_theta(theta_sym))
    nlp.add_ineqconstr(htheta_sym )

    # build optimization variable
    alpha_sym = ca.vertcat(theta_sym, eta_sym)
    nlp.add_var(alpha_sym, alpha0)

    psi = psi_fn(theta=theta_sym, Sigmainv=Sigmainv, x0bar=x0bar)["psi"]
    # build cost
    matrix_cost = ca.mtimes([psi, H, psi.T]) - 2*ca.mtimes([psi, g]) + offset
    nlp.add_cost(0.5 * scaling * ca.trace(matrix_cost) )
    nlp.presolve()
    # store ipopt outputs in a text file, in a dedicated folder
    if not os.path.exists('ipopt_output'):
        os.makedirs('ipopt_output')
    stdout = open('ipopt_output/efficientMethod', 'w')
    with contextlib.redirect_stdout(stdout):
        sol = nlp.solve()

    retrieve = ca.Function("retieve", [nlp.variables], [alpha_sym])
    alpha = retrieve(sol["x"]).full().squeeze()
    status = nlp.status
    cost = nlp.cost / scaling

    return alpha, status, cost

def build_psi_fn(model, m, mpred=1):
    theta_sym = ca.SX.sym("theta", model["dims"]["theta"])
    Sigmainv= ca.SX.sym("Sigmainv", model["dims"]["x"], model["dims"]["x"])
    x0bar = ca.SX.sym("x0bar", model["dims"]["x"])
    Ab_fn, C = build_Phi(model, m - mpred, mpred)
    A, b = Ab_fn(theta_sym, Sigmainv, x0bar) # Phi is defined as A @ Phi + b = 0
    Phi_val = -ca.inv(A) @ b # -ca.solve(A, b)
    psi = C @ Phi_val
    psi_fn = ca.Function("psi_fn", [theta_sym, Sigmainv, x0bar], [psi], ["theta", "Sigmainv", "x0bar"], ["psi"])
    return psi_fn
    
def condense(list_u, list_y, mpred=1):
    N = len(list_y)
    inputs, outputs = build_input_output(list_u, list_y, mpred)
    Hessian = np.einsum("ij,ik->jk", inputs, inputs) / N
    gradient = np.einsum("ij,ik->jk", inputs, outputs) / N
    offset = np.einsum("ij,ik->jk", outputs, outputs) / N
    return Hessian, gradient, offset

def build_input_output(list_u, list_y, mpred):
    m = list_y[0].shape[0]
    mest = m - mpred
    N = len(list_y)
    inputys = np.reshape( list_y[:, :mest,:] , (N, -1))
    inputus = np.reshape( list_u, (N, -1))
    outputs = np.reshape( list_y[:, mest:,:] , (N, -1))
    inputs = np.concatenate([np.ones((N,1)), inputus, inputys], axis=1)
    return inputs, outputs

def build_Phi(model, mest, mpred):
    """
        Transform the problem into the Least-square form
             min_alpha  (y - Phi z)^2
    """
    m = mest + mpred
    cost, variables = build_MHE(model, m) # "theta", "Sigmainv", "x0bar", "us", "ys", "xs"
    y = variables["ys"]
    u = variables["us"]
    xs_flat = ca.reshape(variables["xs"], -1, 1)

    inputy = caflatten(y[:mest,:])
    inputu = caflatten(u)
    input = ca.vertcat(inputu, inputy)
    output = caflatten(y[mest:,:])

    # compute jac
    opti_var = ca.vertcat(xs_flat, output)
    jac = ca.jacobian(cost, opti_var)
    A = ca.jacobian(jac, opti_var)
    b = ca.horzcat(jac.T, ca.jacobian(jac, input))
    C = ca.jacobian(output, opti_var)
    # A Phi + b = 0 to get Phi

    base_var = ca.vertcat(
        variables["theta"],
        ca.reshape(variables["Sigmainv"], -1, 1),
        variables["x0bar"]
    )

    additional = ca.vertcat(opti_var, input)
    zeros = np.zeros(additional.shape[0])
    A_free, b_free = ca.Function("Abfn1", [base_var, additional], [A, b])(base_var, zeros)
    C_free = ca.Function("C", [additional], [C])(zeros)
    Ab_fn = ca.Function("Abfn", [variables["theta"], variables["Sigmainv"], variables["x0bar"]], [A_free, b_free])

    return Ab_fn, C_free

def caflatten(x):
    return ca.vcat([ x[i,:].T for i in range(x.shape[0])])