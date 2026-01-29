import numpy as np
import numpy.linalg as la  
from scipy.linalg import sqrtm

def simulation(model, theta, us, ws, vs, x0):
    N = us.shape[0]
    ytraj = np.zeros((N, model["dims"]["y"]))
    sqrtQ = la.inv( sqrtm(model["Qinv"](theta)) )
    sqrtR = la.inv(sqrtm(model["Rinv"](theta)))
    x = x0.copy()
    for j in range(N):
        x = model["f"](x, us[j], theta) 
        x = x + sqrtQ @ ws[j]
        y = model["g"](x, theta) + sqrtR @ vs[j]
        ytraj[j,:] = y
    return ytraj

def parse(x, jump):
    return x[::jump]

def subsequences(T, Y, m, n_per_traj):
    subTs, subYs = [], []
    for i in range(len(T)):
        subT, subY = [], []
        Ttraj = T[i]
        Ytraj = Y[i]
        N = len(Ttraj)
        step = (N-m) // (n_per_traj-1)
        k_start = 0
        for j in range(n_per_traj):
            subT.append(Ttraj[k_start:k_start+m])
            subY.append(Ytraj[k_start:k_start+m])
            k_start = k_start + step
        subTs.append(subT)
        subYs.append(subY)
    return subTs, subYs

def subsequences2(T, Y, m, step):
    subTs, subYs = [], []
    for i in range(len(T)):
        subT, subY = [], []
        Ttraj = T[i]
        Ytraj = Y[i]
        N = len(Ttraj)
        k_start = 0
        while k_start <= N - m:
            subT.append(Ttraj[k_start:k_start+m])
            subY.append(Ytraj[k_start:k_start+m])
            k_start += step
        subTs.append(subT)
        subYs.append(subY)
    return subTs, subYs
