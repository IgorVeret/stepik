'''Написать функцию для кодирование массива (Run-length encoding). Все подряд идущие повторения элементов функция
сжимает в один элемент и считает количество повторений этого элемента. Функция возвращает кортеж из
двух векторов одинаковой длины. Первый содержит элементы, а второй — сколько раз их нужно повторить.

Пример: encode(np.array([1, 2, 2, 3, 3, 1, 1, 5, 5, 2, 3, 3])) = (np.array[1, 2, 3, 1, 5, 2, 3]),
 np.array[1, 2, 2, 2, 2, 1, 2])'''

import numpy as np


def encode(a):
    is_new_in_row = np.array(a[1:] != a[:-1])
    position = np.append(np.where(is_new_in_row), len(a) - 1)
    lengths = np.diff(np.append(-1, position))
    return (a[position], lengths)

X = np.array([1, 2, 2, 3, 3, 1, 1, 5, 5, 2, 3, 3])

x, num = encode(X)
print(x,num)
