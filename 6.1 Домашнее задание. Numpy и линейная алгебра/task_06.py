import numpy as np

'''Перемножение матриц
Напишите две функции, каждая из которых перемножает две квадратные матрицы:
одна без использования встроенных функций numpy, а другая --- с помощью numpy.
На вход первой задаче подаются списки размера size по size элементов в каждом.
На вход второй задаче подаются объекты типа np.ndarray --- квадратные матрицы одинакового размера. 

Первая функция должна возвращать список списков, а вторая -- np.array.'''


def no_numpy_mult(first, second):
    size = range(len(first[0]))
    return [[sum([first[i][j] * second[j][k] for j in size]) for k in size] for i in size]


def numpy_mult(first, second):
    return np.dot(first, second)
a=[[1,2],[2,1]]
b=[[3,4],[1,3]]
print(no_numpy_mult(a, b))
print(numpy_mult(a, b))