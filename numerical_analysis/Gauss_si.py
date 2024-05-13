def print_reduced_matrix(matrix):
    for row in matrix:
        print(" ".join(map(lambda x: f'{x:.2f}', row[:-1])))

def gauss_elimination(matrix):
    n = len(matrix)

    for i in range(n):
        # Encontre o pivô máximo na coluna atual
        max_row = i
        for j in range(i + 1, n):
            if abs(matrix[j][i]) > abs(matrix[max_row][i]):
                max_row = j
        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]

        # Faça a diagonal principal conter 1
        pivot = matrix[i][i]
        if pivot != 0:
            for j in range(i, n + 1):
                matrix[i][j] /= pivot

        # Reduza as outras linhas
        for j in range(n):
            if j != i:
                factor = matrix[j][i]
                for k in range(i, n + 1):
                    matrix[j][k] -= factor * matrix[i][k]

    return matrix


def main():
    # Matriz estendida [A|B]
    matrix = [
        [1 ,-1, 1, 0],
        [2, -1, 4, 0],
        [3, 1, 11, 0]
    ]

    print("Matriz original:")
    print_reduced_matrix(matrix)

    result = gauss_elimination(matrix)

    print("\nMatriz reduzida:")
    print_reduced_matrix(result)

    print("-------------------------")
    print()

    resulttt = gauss_elimination(matrix)

    print("\nSolução:")
    for row in resulttt:
        print(f"X = {row[-1]:.2f}")

if __name__ == "__main__":
    main()