def f3(x,y):
    def square(x2):
        return x2*x2
    tmp = square(x) + square(y)
    return x,y,tmp