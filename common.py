import numpy as np

def F_vct(F, x):
    vct = []
    for f in F:
        vct.append(f(x))

    return np.array(vct)

def F_norm(F, x):
    return np.linalg.norm(F_vct(F, x))
