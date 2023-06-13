import numpy as np
import numpy.linalg as ln
from svenn_method import svenn
from golden_section_method import golden_section
from dsk_powell_method import dsk_powell
from settings import max_BFGS, schema, h, criterion, method, e_BFGS


f_iter = 0


# given function
def f(x):
    return (100 * (x[0] ** 2 - x[1]) ** 2) + (x[0] - 1) ** 2


# gradient of the initial function
def f_gradient(x):
    if schema == 'central':
        return np.array([(f([x[0] + h, x[1]]) - f([x[0] - h, x[1]])) / (2 * h),
                         (f([x[0], x[1] + h]) - f([x[0], x[1] - h])) / (2 * h)]), 4
    if schema == 'right':
        grad_x_y = f(x)
        return np.array([(f([x[0] + h, x[1]]) - grad_x_y) / h,
                         (f([x[0], x[1] + h]) - grad_x_y) / h]), 3
    if schema == 'left':
        grad_x_y = f(x)
        return np.array([(grad_x_y - f([x[0] - h, x[1]])) / (2 * h),
                         (grad_x_y - f([x[0], x[1] - h])) / (2 * h)]), 3


def one_dimensional_search(a, b, x_k, pk, name):
    if name == 'dsk_powell':
        return dsk_powell(f, a, b, x_k, pk)
    else:
        return golden_section(f, a, b, x_k, pk, 0)


def stop_criterion(grad_fk, x_k, x_k1):
    global f_iter
    if criterion == 'norm':
        return ln.norm(grad_fk) < e_BFGS
    f_xk = f([x_k[0], x_k[1]])
    f_xk1 = f([x_k1[0], x_k1[1]])
    f_iter += 2
    return ln.norm(x_k1 - x_k) / ln.norm(x_k) < e_BFGS and ln.norm(np.abs(f_xk1 - f_xk)) < e_BFGS


def BFGS(f, gradient, x0):
    global f_iter
    # define initial method parameters
    # n_iter = 0     # define var of number of iterations
    grad_fk, g_iter = gradient(x0)    # find the gradient in given point
    f_iter += g_iter
    I = np.eye(len(x0), dtype=int)     # set the identity matrix I
    A_k = I     # set the initial value of A_k of metric matrix equals identity matrix X
    x_k = x0    # set the initial value of x_k equals x_0
    restart_count = 0

    # check the method for the termination criterion
    while True:
        print(x_k)
        pk = -np.dot(A_k, grad_fk)     # find direction of search
        a, b, f_svenn = svenn(f, x_k, pk)
        f_iter += f_svenn
        lambda_k, f_mop = one_dimensional_search(a, b, x_k, pk, method)
        f_iter += f_mop
        print('lambda_k: ', lambda_k)
        if lambda_k == 0:
            A_k = I
            grad_fk, g_iter = gradient(x0)
            f_iter += g_iter
            restart_count += 1
            continue
        x_k1 = x_k + lambda_k * pk    # find the next point x_k+1 by recurrent formula

        delta_xk = x_k1 - x_k     # find the value of algorithm step (delta_x_k)
        if stop_criterion(grad_fk, x_k, x_k1):
            return x_k, restart_count

        x_k = x_k1     # update point value
        grad_fk1, g_iter = gradient(x_k1)    # update gradient of function in new point x_k+1
        f_iter += g_iter
        delta_grad_fk = grad_fk1 - grad_fk    # find delta gradient
        grad_fk = grad_fk1
        # n_iter += 1     # update number of iterations

        # correcting matrix has formula: A_k+1 = [I - term1] * A_k * [I - term2] + term3
        term1 = I - (delta_xk[:, np.newaxis] * delta_grad_fk[np.newaxis, :]) / np.dot(delta_xk, delta_grad_fk)
        term2 = I - (delta_xk[np.newaxis, :]) * delta_grad_fk[:, np.newaxis] / np.dot(delta_xk, delta_grad_fk)
        term3 = (delta_xk[:, np.newaxis] * delta_xk[np.newaxis, :]) / np.dot(delta_xk, delta_grad_fk)
        A_k = np.dot(term1, np.dot(A_k, term2)) + term3


result, r_count = BFGS(f, f_gradient, np.array([-1.2, 0]))
print('Point: ', result)
print('f(x) = ', f(result))
print('r_count: ', r_count)
print('f_iter: ', f_iter)
