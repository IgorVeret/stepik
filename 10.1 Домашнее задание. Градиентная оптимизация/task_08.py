'''Еще один градиент, похожий на тот, что был на лекции:
ψ(x,y,z)=ln(cos(ex+y))−ln(xy)
Заполните пропуски в функции ниже'''
from math import sin, cos, tan, exp, pi
import numpy as np

def grad_2(x, y, z):
    # возвращает кортеж из 3 чисел --- частных производных по x,y,z
    dx = -(x * np.e ** (x + y) * np.tan(np.e ** (x + y)) + 1) / x
    dy = -(y * np.e ** (x + y) * np.tan(np.e ** (x + y)) + 1) / y
    dz = 0

    return (dx, dy, dz)

#Тестируем нашу функцию
for i in grad_2(1,1,0), grad_2(-10, 3, 0), grad_2(15 ,4, 0):
    print('Результаты', i)

assert np.allclose(grad_2(1,1,0), (-15.73101919885423, -15.73101919885423, 0), atol=5e-6)
assert np.allclose(grad_2(-10, 3, 0), (0.09999916847105042, -0.3333341648622829, 0), atol=5e-6)
assert np.allclose(grad_2(15 ,4, 0), (54654806.79650013, 54654806.6131668,0), atol=5e-6)