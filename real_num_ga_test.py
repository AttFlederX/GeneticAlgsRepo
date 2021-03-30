from algs.real_num_genetic_alg import RealNumberGeneticAlgorithm

import numpy as np
import datetime

# Default test
print('\nReal number GA, default test')
solver = RealNumberGeneticAlgorithm(
    a=-1,
    b=2,
    f=lambda x: x * np.sin(10 * np.pi * x) + 1,
    prec=6
)

start = datetime.datetime.now()
res, x, t = solver.solve(verbose_mode=True, graphing_mode=True)
end = datetime.datetime.now()

print(f'Result: {res} at x={x}, {t + 1} iterations')
print(f'Time: {(end - start).total_seconds()} seconds')

input()

# n_p test
print('\nReal number, n_p test')

N_p = [10, 20, 50, 100, 200, 500, 1000]

for n_p in N_p:
    print(f'\n\tn_p = {n_p}: \n')
    solver = RealNumberGeneticAlgorithm(
        a=-1,
        b=2,
        f=lambda x: x * np.sin(10 * np.pi * x) + 1,
        prec=6,
        
        n_p=n_p
    )

    start = datetime.datetime.now()
    res, x, t = solver.solve()
    end = datetime.datetime.now()

    print(f'Result: {res} at x={x}, {t + 1} iterations')
    print(f'Time: {(end - start).total_seconds()} seconds')

input()


# n_t test
print('\nReal number, n_t test')

N_t = [2, 3, 4, 5, 6]

for n_t in N_t:
    print(f'\n\tn_t = {n_t}: \n')
    solver = RealNumberGeneticAlgorithm(
        a=-1,
        b=2,
        f=lambda x: x * np.sin(10 * np.pi * x) + 1,
        prec=6,
        
        n_t=n_t
    )

    start = datetime.datetime.now()
    res, x, t = solver.solve()
    end = datetime.datetime.now()

    print(f'Result: {res} at x={x}, {t + 1} iterations')
    print(f'Time: {(end - start).total_seconds()} seconds')

input()


# p_c test
print('\nReal number, p_c test')

P_c = [1, 0.9, 0.8, 0.5, 0.1]

for p_c in P_c:
    print(f'\n\tp_c = {p_c}: \n')
    solver = RealNumberGeneticAlgorithm(
        a=-1,
        b=2,
        f=lambda x: x * np.sin(10 * np.pi * x) + 1,
        prec=6,
        
        p_c=p_c
    )

    start = datetime.datetime.now()
    res, x, t = solver.solve()
    end = datetime.datetime.now()

    print(f'Result: {res} at x={x}, {t + 1} iterations')
    print(f'Time: {(end - start).total_seconds()} seconds')

input()