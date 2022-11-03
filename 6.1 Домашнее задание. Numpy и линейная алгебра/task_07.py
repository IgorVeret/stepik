'''Вам подаются на вход два вектора a и b в трехмерном пространстве.
Реализуйте их скалярное произведение с помощью numpy и без. '''
import numpy as np


def no_numpy_scalar(v1, v2):
    # param v1, v2: lists of 3 ints
    return sum([ai * bi for ai, bi in zip(v1, v2)])


def numpy_scalar(v1, v2):
    # param v1, v2: np.arrays[3]
    return np.dot(v1, v2)


a = [1, 2, 1]
b = [3, 4, 3]
print(no_numpy_scalar(a, b))
print(numpy_scalar(a, b))
