import numpy as np
from settings import e_DSK


def dsk_powell_iteration(interval):
    values = np.array(list(interval.values()))
    center = list(interval.keys())[np.where(values[:, 1] == min(values[:, 1]))[0][0]]
    s_interval = sorted(interval.items(), key=lambda x: x[1][0])
    s_interval = {s_interval[i][0]: s_interval[i][1] for i in range(len(s_interval))}

    center_index = list(s_interval.keys()).index(center)
    if center_index == len(s_interval) - 1:
        return s_interval[center][0]
    x1, x3 = list(s_interval.keys())[center_index - 1], list(s_interval.keys())[center_index + 1]
    new_interval = {'x1': s_interval[x1], 'x2': s_interval[center], 'x3': s_interval[x3]}
    return new_interval


def dsk_powell(f, x1, x3, x0, pk):
    f_iter = 0
    x2 = (x1 + x3) / 2
    delta_x = x2 - x1
    f_x1, f_x2, f_x3 = [f([x0[0] + i * pk[0], x0[1] + i * pk[1]]) for i in [x1, x2, x3]]
    x = x2 + (delta_x * (f_x1 - f_x3)) / (2 * (f_x1 - 2 * f_x2 + f_x3))
    f_x = f([x0[0] + x * pk[0], x0[1] + x * pk[1]])
    f_iter += 4

    if np.abs(f_x2 - f_x) <= e_DSK and np.abs(x2 - x) <= e_DSK:
        return x, f_iter

    while True:
        interval = dsk_powell_iteration({'x1': (x1, f_x1), 'x2': (x2, f_x2), 'x3': (x3, f_x3), 'x*': (x, f_x)})
        if not isinstance(interval, dict):
            return interval, f_iter
        a1 = (interval['x2'][1] - interval['x1'][1]) / (interval['x2'][0] - interval['x1'][0])
        a2 = (((interval['x3'][1] - interval['x1'][1]) / (interval['x3'][0] - interval['x1'][0])) -
              ((interval['x2'][1] - interval['x1'][1]) / (interval['x2'][0] - interval['x1'][0]))) / (interval['x3'][0] - interval['x2'][0])
        x = (interval['x1'][0] + interval['x2'][0]) / 2 - a1 / (2 * a2)
        f_x = f([x0[0] + x * pk[0], x0[1] + x * pk[1]])
        f_iter += 1
        if np.abs(f_x2 - f_x) <= e_DSK and np.abs(x2 - x) <= e_DSK:
            return x, f_iter
        x1, x2, x3 = [interval[i][0] for i in ['x1', 'x2', 'x3']]
        f_x1, f_x2, f_x3 = [interval[i][1] for i in ['x1', 'x2', 'x3']]
