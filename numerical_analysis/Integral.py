import numpy as np

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    integral = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        integral += f(a + i * h)
    integral *= h
    return integral

def f(t):
    return 1 / t

# Parâmetros de precisão
tolerance = 1e-4

def calculate_ln(x):
    n = 1
    error = 1
    previous_result = 0
    iterations = 0

    while error > tolerance:
        result = trapezoidal_rule(f, 1, x, n)
        error = abs(result - previous_result) / abs(result)
        previous_result = result
        n *= 2
        iterations += 1

    return result, n, iterations

# Cálculo do ln(17)
ln_17, n_17, iterations_17 = calculate_ln(17)
print(f"ln(17) = {ln_17} com {n_17} pontos e {iterations_17} iterações")

# Cálculo do ln(35)
ln_35, n_35, iterations_35 = calculate_ln(35)
print(f"ln(35) = {ln_35} com {n_35} pontos e {iterations_35} iterações")

