import numpy as np


# lambda expression for the f(x)

#f = lambda x : (x**4)* np.exp(-x**2)*np.sin(x)

f = lambda x : np.log10(np.sqrt(x**2 + x) + 1)

def simpson38( a , b , x , n):
    
    print("Simpson 3/8")

    
    if n % 3 != 0:
        raise ValueError("n must be a multiple of 3")
    
    h = (b - a) / n
    integral = f(a) + f(b)

    for i in range(1,n):
        if i % 3 == 0:
            integral += 2 * f(a + i * h)
        else:
            integral += 3 * f(a + i * h)
            
    integral = 3 * h * integral / 8
    
    return integral

    
def trapezio(a , b , f ,n):
    h = (b-a)/n
    
    x = np.linspace(a, b, n+1)
    
    y=f(x)

    integral = (h/2) * (y[0] + 2*np.sum(y[1:n]) + y[n])

    return integral

def erro_trapezoidal(f ,a , b ,tol):
    n = 1
    integral_prev = trapezio(a , b , f ,n)

    while True:
        n = n*2
        integral = trapezio(a , b , f ,n)

        if abs(integral - integral_prev) < tol:
            return integral
        integral_prev = integral

a = 2

b = 50

n = 60

tol = 1e-10

#res = simpson38(a, b, f, n)

res = erro_trapezoidal(f, a, b, tol)

print("\n\n\n\ Integral: ", res)
    
    #todo: method