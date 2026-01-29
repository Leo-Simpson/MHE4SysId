import numpy as np

def prepare_data(U, Y, m, step=1):
    list_u = []
    list_y = []
    ntraj = len(U)
    for i in range(ntraj):
        N = len(U[i])
        j = 0
        while j <= N - m:
            u = U[i][j:j+m]
            y = Y[i][j:j+m]
            list_u.append(u)
            list_y.append(y)
            j += step
    list_u = np.array(list_u)
    list_y = np.array(list_y)
    return list_u, list_y

def resample(list_u, list_y, rng, N):
    Ntot = len(list_u)
    idx = rng.choice(Ntot, N, replace=False)
    list_u_small = list_u[idx, :,:]
    list_y_small = list_y[idx,:,:]
    return list_u_small, list_y_small
