'''На вход дан двумерный массив X
 Выходом функции должен быть двумерный массив той же формы, что и X.'''

import numpy as np
import scipy.stats as sps


def cumsum(A):
    # param A: np.array[m,n]
    # YOUR CODE

    result = np.cumsum(A, axis=0)
    # result = np.cumsum(A, axis=1)
    return result

# зададим некоторую последовательность и проверим ее на вашей функции.
A = sps.uniform.rvs(size=10**3)

S2 = cumsum(A)
print(S2)