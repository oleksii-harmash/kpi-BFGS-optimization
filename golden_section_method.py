from settings import e_GS


def golden_section(f, a, b, x, pk, f_iter):
    if b - a <= e_GS:
        return (a + b) / 2, f_iter
    x1 = a + 0.382 * (b - a)
    x2 = a + 0.618 * (b - a)
    f_x1 = f([x[0] + x1 * pk[0], x[1] + x1 * pk[1]])
    f_x2 = f([x[0] + x2 * pk[0], x[1] + x2 * pk[1]])
    if f_x1 < f_x2:
        return golden_section(f, a, x2, x, pk, f_iter + 2)
    else:
        return golden_section(f, x1, b, x, pk, f_iter + 2)
