import numpy as np
import casadi as ca

class NLP:
    def __init__(self):
        self.g = []
        self.lbg = []
        self.ubg = []
        self.cost = 0.
        self.variables = []
        self.warm_starts = []
        self.lbx = []
        self.ubx = []
    
    def add_var(self, var, ws, lb=-np.inf, ub=np.inf):
        var_flat = ca.reshape(var, -1, 1)
        ws_flat = ca.reshape(ws, -1, 1)
        self.variables = ca.vertcat(self.variables, var_flat)
        self.warm_starts = ca.vertcat(self.warm_starts, ws_flat)
        dim = var_flat.shape[0]
        lb_vect = lb * np.ones(dim)
        ub_vect = ub * np.ones(dim)
        self.lbx = ca.vertcat(self.lbx, lb_vect)
        self.ubx = ca.vertcat(self.ubx, ub_vect)
        return var
    
    def add_eqconstr(self, constr):
        constr_flat = ca.reshape(constr, -1, 1)
        n = constr_flat.shape[0]
        self.g = ca.vertcat(self.g, constr_flat)
        self.lbg = ca.vertcat(self.lbg, np.zeros(n))
        self.ubg = ca.vertcat(self.ubg, np.zeros(n))
    
    def add_ineqconstr(self, constr):
        constr_flat = ca.reshape(constr, -1, 1)
        n = constr_flat.shape[0]
        self.g = ca.vertcat(self.g, constr_flat)
        self.lbg = ca.vertcat(self.lbg, -np.inf*np.ones(n))
        self.ubg = ca.vertcat(self.ubg, np.zeros(n))
    
    def add_cost(self, stage_cost):
        self.cost += stage_cost

    def _solver_options(self, opts) -> dict:
        self.nlpsolver_options = {}
        self.nlpsolver_options["expand"] = False
        # self.nlpsolver_options["ipopt.min_hessian_perturbation"] = 1.
        self.nlpsolver_options["ipopt.max_iter"] = 500
        self.nlpsolver_options["ipopt.max_cpu_time"] = 3600.0
        self.nlpsolver_options["ipopt.linear_solver"] = "mumps"  # suggested: ma57
        self.nlpsolver_options["ipopt.mumps_mem_percent"] = 10000
        self.nlpsolver_options["ipopt.mumps_pivtol"] = 0.001
        self.nlpsolver_options["ipopt.print_level"] = 5 # or 3
        self.nlpsolver_options["ipopt.print_frequency_iter"] = 10

        for key, value in opts.items():
            self.nlpsolver_options[key] = value

    def presolve(self, opts={}):
        self._solver_options(opts)
        nlp = {"x": self.variables, "f": self.cost, "g": self.g}
        self.nlpsol = ca.nlpsol("S", "ipopt", nlp, self.nlpsolver_options)

    def solve(self) -> tuple:
        r = self.nlpsol(
            x0=self.warm_starts,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
        )
        self.status = self.nlpsol.stats()["return_status"]
        self.cost = r["f"].full().squeeze()
        return r
