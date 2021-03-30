from algs.multivar_genetic_alg import MultiVariableGeneticAlgorithm

import numpy as np
import datetime

# Default test
print('\nMulti-variable GA, default test')
solver = MultiVariableGeneticAlgorithm(
    a=[0, -2],
    b=[7, 6],
    f=[
        lambda x: -13 + x[0] + ((5 - x[1])*x[1] - 2) * x[1],
        lambda x: -29 + x[0] + ((x[1] + 1)*x[1] - 14) * x[1],
    ],
    prec=6
)

start = datetime.datetime.now()
res, x, t = solver.solve(verbose_mode=True, graphing_mode=True)
end = datetime.datetime.now()

print(f'Result: {res} at x={x}, {t + 1} iterations')
print(f'Time: {(end - start).total_seconds()} seconds')

input()

# n_p test
print('\nMulti-variable GA, n_p test')

N_p = [10, 20, 50, 100, 200, 500, 1000]

for n_p in N_p:
    print(f'\n\tn_p = {n_p}:')
    solver = MultiVariableGeneticAlgorithm(
        a=[0, -2],
        b=[7, 6],
        f=[
            lambda x: -13 + x[0] + ((5 - x[1])*x[1] - 2) * x[1],
            lambda x: -29 + x[0] + ((x[1] + 1)*x[1] - 14) * x[1],
        ],
        prec=6,
        
        n_p=n_p
    )

    start = datetime.datetime.now()
    res, x, t = solver.solve()
    end = datetime.datetime.now()

    print(f'Result: {res} at x={x}, {t + 1} iterations')
    print(f'Time: {(end - start).total_seconds()} seconds')

input()


# p_e test
print('\nMulti-variable GA, p_e test')

P_e = [0.1, 0.2, 0.3, 0.4, 0.5]

for p_e in P_e:
    print(f'\n\tp_e = {p_e}:')
    solver = MultiVariableGeneticAlgorithm(
        a=[0, -2],
        b=[7, 6],
        f=[
            lambda x: -13 + x[0] + ((5 - x[1])*x[1] - 2) * x[1],
            lambda x: -29 + x[0] + ((x[1] + 1)*x[1] - 14) * x[1],
        ],
        prec=6,
        
        p_e=p_e
    )

    start = datetime.datetime.now()
    res, x, t = solver.solve()
    end = datetime.datetime.now()

    print(f'Result: {res} at x={x}, {t + 1} iterations')
    print(f'Time: {(end - start).total_seconds()} seconds')

input()


# r test
print('\nMulti-variable GA, r test')

R = [0.05, 0.1, 0.2, 0.3, 0.45]

for r in R:
    print(f'\n\tr = {r}:')
    solver = MultiVariableGeneticAlgorithm(
        a=[0, -2],
        b=[7, 6],
        f=[
            lambda x: -13 + x[0] + ((5 - x[1])*x[1] - 2) * x[1],
            lambda x: -29 + x[0] + ((x[1] + 1)*x[1] - 14) * x[1],
        ],
        prec=6,
        
        r=r
    )

    start = datetime.datetime.now()
    res, x, t = solver.solve()
    end = datetime.datetime.now()

    print(f'Result: {res} at x={x}, {t + 1} iterations')
    print(f'Time: {(end - start).total_seconds()} seconds')

input()