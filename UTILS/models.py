import numpy as np

def simpleModel(n=1):
    # theta = [a, b]
    dims = {
        "x":n, "y":n, "u":n, "theta":2
    }

    def f(x, u, theta, vect_type=np.array):
        a = theta[0]
        b = theta[1]
        return a * x + (1-a) * b * u
    
    def g(x, theta):
        return x
    
    def Qinv(theta):
        return np.eye(dims["x"])
    
    def Rinv(theta):
        return np.eye(dims["y"])

    def h(theta):
        # inequality constraints on the form h(theta) <= 0
        a = theta[0]
        b = theta[1]

        amin, amax = 0.1, 1.
        bmin, bmax = 0.1,  1.
        hlist = [
                amin - a,
                bmin - b,
                a - amax,
                b - bmax
            ]
        return hlist

    return {"f": f, "g": g, "h":h, "Qinv": Qinv, "Rinv": Rinv,
            "dims": dims}

def counterExModel(Q=1e-3):
    dims = {
        "x":1, "y":1, "u":1, "theta":1
    }

    def f(x, u, theta, vect_type=np.array):
        return theta * x
    
    def g(x, theta):
        return 1. * x
    
    def Qinv(theta):
        return np.eye(1) / Q
    
    def Rinv(theta):
        return np.eye(1)

    def h(theta):
        # inequality constraints on the form h(theta) <= 0
        a = theta[0]
        amin, amax = 0., 1.
        hlist = [
                amin - a,
                a - amax,
            ]
        return hlist
    return {"f": f, "g": g, "h":h, "Qinv": Qinv, "Rinv": Rinv,
            "dims": dims}

def LorenzModel(dt, jump=1, measure=[1], beta=None):
    # theta = [sigma, rho, beta]
    dims = {
        "x":3, "y":len(measure), "u":1
    }
    if beta is None:
        dims["theta"] = 3
    else:
        dims["theta"] = 2
    
    def fcontinuous(x, theta, vect_type=np.array):
        sigma = theta[0]
        rho = theta[1]
        if beta is None:
            beta_ = theta[2]
        else:
            beta_ = beta

        x1 = x[0]
        x2 = x[1]
        x3 = x[2]

        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta_ * x3
        dx = [dx1, dx2, dx3]
        dx = vect_type(dx)
        return dx
    
    def RK4(x, theta, step, vect_type=np.array):
        k1 = fcontinuous(x, theta, vect_type=vect_type)
        k2 = fcontinuous(x + step/2 * k1, theta, vect_type=vect_type)
        k3 = fcontinuous(x + step/2 * k2, theta, vect_type=vect_type)
        k4 = fcontinuous(x + step * k3, theta, vect_type=vect_type)
        xnew = x + step/6 * (k1 + 2*k2 + 2*k3 + k4)
        return xnew
    
    def Euler(x, theta, step, vect_type=np.array):
        dx = fcontinuous(x, theta, vect_type=vect_type)
        xnew = x + step * dx
        return xnew
    
    def f(x, u, theta, vect_type=np.array):
        x_ = 1. * x
        for i in range(jump):
            x_ = RK4(x_, theta, dt, vect_type=vect_type)
        return x_
    
    def g(x, theta):
        return x[measure]
    
    def Qinv(theta):
        return np.eye(dims["x"]) * 1.
    
    def Rinv(theta):
        return np.eye(dims["y"]) * 1.

    def h(theta):
        # inequality constraints on the form h(theta) <= 0
        thetamin = 1. * np.ones(dims["theta"])
        thetamax = 100. * np.ones(dims["theta"])
        hlist = [
                thetamin - theta,
                theta - thetamax
            ]
        return hlist

    return {"f": f, "g": g, "h":h, "Qinv": Qinv, "Rinv": Rinv,
            "dims": dims}

def OscillatorModel(dt, jump=1):
    # theta = [omega]
    dims = {
        "x":2, "y":1, "u":1, "theta":1
    }
    
    def fcontinuous(x, theta): 
        omega = theta[0]
        A = np.array([[0, 1], [-omega**2, 0]])
        dx = A @ x
        return dx
    
    def rk4(x, theta, step):
        k1 = fcontinuous(x, theta)
        k2 = fcontinuous(x + step/2 * k1, theta)
        k3 = fcontinuous(x + step/2 * k2, theta)
        k4 = fcontinuous(x + step * k3, theta)
        xnew = x + step/6 * (k1 + 2*k2 + 2*k3 + k4)
        return xnew
    
    def f(x, u, theta):
        x_ = 1. * x
        for i in range(jump):
            x_ = rk4(x_, theta, dt)
        return x_
    
    def g(x, theta):
        return x[0]
    
    def Qinv(theta):
        return np.eye(dims["x"]) * 1e2
    
    def Rinv(theta):
        return np.eye(dims["y"]) * 1.

    def h(theta):
        # inequality constraints on the form h(theta) <= 0
        hlist = [
            ]
        return hlist

    return {"f": f, "g": g, "h":h, "Qinv": Qinv, "Rinv": Rinv,
            "dims": dims}