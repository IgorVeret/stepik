'''Вычислите производную  f(x)=tg(x)⋅ln(cos(x2)+1) , в точке  x0=0 . Ответ округлите до двух знаков после запятой.'''
import  numpy as np
x = np.linspace(-0.00001, 0.00001, 3)
dx = x[1] - x[0]
y = np.tan(x) * np.log(np.cos(x**2) + 1)
dydx = np.gradient(y, dx)
print(round(dydx[1],2))