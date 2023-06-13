from numpy.linalg import norm
from settings import init_lambda, delta_cf


def svenn(f, x, pk):
    f_iter = 0
    lambdas = [init_lambda]
    f_lambdas = [f(x)]
    delta_lambda = delta_cf * norm(x) / norm(pk)

    f_plus_delta = f(x + (lambdas[0] + delta_lambda) * pk)
    f_minus_delta = f(x + (lambdas[0] - delta_lambda) * pk)

    if f_plus_delta > f_minus_delta:
        f_lambdas.append(f_minus_delta)
        delta_lambda = -1 * delta_lambda
    else:
        f_lambdas.append(f_plus_delta)

    lambdas.append(lambdas[0] + delta_lambda)
    f_iter += 3

    while True:
        f_iter += 1
        delta_lambda *= 2
        lambdas.append(lambdas[-1] + delta_lambda)
        f_lambdas.append(f(x + lambdas[-1] * pk))
        if f_lambdas[-1] > f_lambdas[-2]:
            lambdas.append((lambdas[-1] + lambdas[-2]) / 2)
            f_lambdas.append(f(x + lambdas[-1] * pk))
            return lambdas[-4], lambdas[-1], f_iter
