def almost_double_factorial(n):
    item = 1
    for i in range(n + 1):

        if i % 2 > 0:
            item *= i

    return item
if __name__ == "__main__":
    x=almost_double_factorial(10)
    print(x)