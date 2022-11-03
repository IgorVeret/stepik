'''Пример работы функции:

A = [5, 1, 4, 5, 14]
B = cumsum_and_erase(A, erase=10)
assert B == [5, 6, 15, 29], "Something is wrong! Please try again"'''
def cumsum_and_erase(A, erase=1):
    result=[]
    z=0
    for i in A:
        z = z + i
        if z==erase:
            continue
        else:
            result.append(z)

    B = result
    return B