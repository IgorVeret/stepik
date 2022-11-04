'''Посчитайте производную  f(x)=xx  в точке  x0=e
Ответ округлите до одного знака после запятой.
Указание. Представьте функцию  f(x)  как  eg(x)  для некоторой  g .'''
from sympy import symbols, diff

from math import e

x = symbols('x')

function = pow(x, x)

df_f = diff(function, x)

print(round(df_f.subs(dict(x=e)),  1))