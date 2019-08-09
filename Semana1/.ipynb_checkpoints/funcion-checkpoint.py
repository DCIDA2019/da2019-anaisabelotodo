def f3(x,y):
    def cuadrado(x2):
        return x2*x2
    tmp = cuadrado(x) + cuadrado(y)
    return x,y,tmp