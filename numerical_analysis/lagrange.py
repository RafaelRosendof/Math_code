def lagrange_interpolation(x, y, xi):
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (xi - x[j]) / (x[i] - x[j])
        result += term
    return result

# Exemplo de uso
x = [1, 2, 4]
y = [3, 5, 7]
xi = 3
result = lagrange_interpolation(x, y, xi)
print("Para x =", xi, "o valor interpolado é:", result)
