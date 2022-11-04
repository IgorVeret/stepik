"""В лекции было несколько функций, чьи градиенты Вам было предложено вычислить.
Вычислите градиент следующей функции:
ψ(x,y,z)=sin(xz)−y2z+ex
Заполните пропуски в коде"""
from math import sin, cos, tan, exp, pi

import numpy as np


def grad_1(x, y, z):
    # возвращает кортеж из 3 чисел --- частных производных по x,y,z
    dx = np.e ** x + z * np.cos(x * z)
    dy = -2 * y * z
    dz = x * cos(x * z) - y ** 2
    return (dx, dy, dz)


# Тестируем нашу функцию
for i in grad_1(1, 1, 1), grad_1(1, 8, 0), grad_1(-11, pi, 1):
    print('Результаты', i)

assert np.allclose(grad_1(1, 1, 1), (3.258584134327185, -2, -0.45969769413186023), atol=5e-6)
assert np.allclose(grad_1(1, 8, 0), (2.718281828459045, 0, -63.0), atol=5e-6)
assert np.allclose(grad_1(-11, pi, 1), (0.004442399688841031, -6.283185307179586, -9.918287078957917), atol=5e-6)
