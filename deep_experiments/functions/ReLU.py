#Code inpired by the work of nikolaj-K in the youtube 
#I'm using this code for my personal study in neural networks implementation and mathematics to
'''
Author: Rafael
Date: 19/01/2024
Abstract: Calculate polynomials with ReLU
'''
import matplotlib.pyplot as plt
import numpy as np
import torch


class Cfg:
    SCALE = 1 / 4
    xs = [x / 300 for x in range(300)]
    PLOT_RANGE = [0, 1]  # For both x- and y-axis


def iterate(f, n, x): # f função , n iteração , x o input
    y = x
    for _ in range(n):
        y = f(y)
    return y


def square(x):
    return x**2


def bump(x):
    return x * (1 - x)


def neumann_series(func_eval_pow, n_max):
    return sum(map(func_eval_pow, range(n_max)))


def ReLU(x):
    # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    return max(0, x)#ativação da Relu


def ReLU_vec(xs):
    return np.array(list(map(ReLU, xs)))


class Triangle: #representa uma função de onda triangular 
    # https://en.wikipedia.org/wiki/Triangle_wave
    # Here: Only need to consider in x in the interval Cfg.PLOT_RANGE
    def __init__(self, left_end=0, right_end=1, slope=2):
        self.SLOPES = [slope, slope * -2]  # Twice as strongly down as up
        center = (right_end - left_end) / 2
        self.OFFSETS = [-left_end, -center]

    def eval(self, x):
        return self.SLOPES @ ReLU_vec([x + d for d in self.OFFSETS])


def triangle_sum(neumann_n_max, x): #calula a série de neuman para uma função triangular 
    tri = Triangle()
    t = tri.eval(x)
    def triangle_iter_at_t(n):
        DENOM = 4
        return iterate(tri.eval, n, t) / DENOM**n
    return neumann_series(triangle_iter_at_t, neumann_n_max)


def neumann_square(neumann_n_max, x):
    return x - Cfg.SCALE * triangle_sum(neumann_n_max, x)


def plot_1():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ys_ref = np.array(list(map(bump, Cfg.xs)))  # Note: 4 x * (1 - x) = 1 - (2 * x - 1)**2

    NUM_SUMMANDS = 20

    ITER_TRI_PLOT = (0,)  # Red dashed lines in plot

    # Actual script:
    tri = Triangle()
    DENOMS = (1.5, 1.7, 2, 2.5, 3, 4, 7)
    # The 2 gives the nowhere-differentiable but uniformly continuous Takagi curve, see below
    # The 4 gives the square
    ys_sums = {d: np.zeros(len(Cfg.xs)) for d in DENOMS}
    for n in range(NUM_SUMMANDS):
        ys = np.array([iterate(tri.eval, n, tri.eval(x)) for x in Cfg.xs])
        if n in ITER_TRI_PLOT:
            ax.plot(Cfg.xs, Cfg.SCALE * ys, color='tab:red', linestyle='dashed')

        for d in DENOMS:
            ys_sums[d] += ys / d**n

    # Plot sums
    for d, ys_sum in ys_sums.items():
        # https://en.wikipedia.org/wiki/Blancmange_curve
        # https://en.wikipedia.org/wiki/Teiji_Takagi
        # Note: My scaling are tuned for squaring,
        # so might not matched the Wikipedia formulas for tri and iterate(triangle
        ax.plot(Cfg.xs, Cfg.SCALE * ys_sum, linewidth=1, label=d)

    # Plot quadratic reference curve (black dots)
    ax.plot(Cfg.xs, ys_ref, color='black', linestyle='dotted', linewidth=2.5)

    ax.set_xlim(Cfg.PLOT_RANGE)
    ax.set_ylim(Cfg.PLOT_RANGE)
    ax.legend(loc='upper right')
    plt.gca().set_aspect('equal')
    plt.show()


class LayerWeights: #definindo os pesos e biases para as camadas da rede neural 
    def __init__(self, num_layers):
        def arr(xs):
            return np.array(xs, dtype=np.double)

        tri = Triangle()
        B = arr(tri.OFFSETS + [0])  # Extend with iteration sum channel mask
        W = arr(tri.SLOPES + [0])
        def w(h):
            DENOM = 4
            slopes = W / DENOM**h
            slopes[2] += 1  # Extend with iteration sum channel
            return slopes

        self.B_IN = B
        self.W_IN = arr([1, 1, 0])
        # Triangle mask, giving [1*x-0, 1*x-1/2, 0], then ReLU_vec is applied

        self.B_H = B  # Also always -1/2 for the second value
        self.w_hs = [[W, W, w(h)] for h in range(1, num_layers)]
        # Only in the last row, and only in the first and second coponent, do we have non-constant weights

        self.B_OUT = arr([0])
        self.w_out = w(num_layers)  # Vector for final scalar product of all data


def manual_nn_bump(num_summands, xs, log=False):
    lw = LayerWeights(num_summands)

    for x in xs:
        ys = ReLU_vec(lw.W_IN * x + lw.B_IN)
        for idx, wh in enumerate(lw.w_hs):
            ys = ReLU_vec(wh @ ys + lw.B_H)
            if log:
                print(f"x={round(x, 3)}, camada oculta y's #{idx}: {ys}")

        yield ReLU(ys @ lw.w_out + lw.B_OUT[0])
        
        '''
        Nessa função, a rede neural é implementada manualmente. Ela possui uma camada de entrada, 
        camadas ocultas e uma camada de saída. As ativações ReLU são aplicadas após cada camada.
        A saída da rede é obtida através do produto escalar dos pesos finais e da adição do bias final.
        '''


def pytorch_nn_bump(depth, xs):
    IN, HIDDEN, OUT = 1, 3, 1 #camadas de entrada, escondidas e de saída

    lw = LayerWeights(depth)

    l = torch.nn.Linear(IN, HIDDEN)
    l.bias.data = torch.nn.Parameter(torch.tensor(lw.B_IN))
    l.weight.data = torch.nn.Parameter(torch.tensor(lw.W_IN).view(HIDDEN, IN))
    layers = [l, torch.nn.ReLU()]

    for wh in lw.w_hs:
        l = torch.nn.Linear(HIDDEN, HIDDEN)
        l.bias.data = torch.nn.Parameter(torch.tensor(lw.B_H))
        l.weight.data = torch.nn.Parameter(torch.tensor(wh))
        layers += [l, torch.nn.ReLU()]
        
        '''
        Aqui, a rede neural é implementada utilizando PyTorch. São criadas camadas lineares (torch.nn.Linear)
        e funções de ativação ReLU. A rede completa é montada usando torch.nn.Sequential. 
        A função model_scalar recebe um valor de entrada, aplica a rede, e retorna a saída como um número NumPy.

    Essas duas funções (manual_nn_bump e pytorch_nn_bump) são usadas para calcular as saídas das 
    redes para diferentes valores de entrada, que são posteriormente plotadas para visualização 
        '''

    l = torch.nn.Linear(HIDDEN, OUT)
    l.bias.data = torch.nn.Parameter(torch.tensor(lw.B_OUT))
    l.weight.data = torch.nn.Parameter(torch.tensor(lw.w_out).view(OUT, HIDDEN))
    layers += [l, torch.nn.ReLU()]

    model = torch.nn.Sequential(*layers)

    def model_scalar(x):  # Wrap scalar input and unwrap to scalar output
        x_ = torch.DoubleTensor([x])  # DoubleTensor in accordante with dtype=np.double used above
        y_ = model(x_)
        return y_.data.numpy()[0]

    for x in xs:
        yield model_scalar(x)


def plot_2():
    ys_ref = list(map(bump, Cfg.xs))

    NUM_SUMMANDS = 4  # = pytorch network depth
    ys_manual = list(manual_nn_bump(NUM_SUMMANDS, Cfg.xs, log=True))
    ys_torch = list(pytorch_nn_bump(NUM_SUMMANDS, Cfg.xs))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(Cfg.xs, ys_ref, color='black', linestyle='dotted', linewidth=1.5)
    ax.plot(Cfg.xs, ys_manual, color='red', linestyle='dashed', linewidth=2.5)
    ax.plot(Cfg.xs, ys_torch, color='green', linestyle='solid', linewidth=1.5)

    def just_y_squared(zs):
        assert len(Cfg.xs) == len(zs)
        return [x - z for x, z in zip(Cfg.xs, zs)]
    ax.plot(Cfg.xs, just_y_squared(ys_ref), color='black', linestyle='dotted', linewidth=1.5)
    ax.plot(Cfg.xs, just_y_squared(ys_manual), color='red', linestyle='dashed', linewidth=2.5)
    ax.plot(Cfg.xs, just_y_squared(ys_torch), color='green', linestyle='solid', linewidth=1.5)

    ax.set_xlim(Cfg.PLOT_RANGE)
    ax.set_ylim(Cfg.PLOT_RANGE)
    plt.gca().set_aspect('equal')
    plt.show()

# performs assertions to check if the neural network approximations match the expected results
def runs_square_asserts():
    def square_function(f, num_summands, x):
        xs = [x]
        ys = list(f(num_summands, xs))
        return x - ys[0]

    NUM_TERMS = 20
    for idx, x in enumerate(Cfg.xs):
        x_squared = square(x)
        neumann_delta = neumann_square(NUM_TERMS, x) - x_squared
        manual_nn_delta = square_function(manual_nn_bump, NUM_TERMS, x) - x_squared
        torch_nn_delta = square_function(pytorch_nn_bump, NUM_TERMS, x) - x_squared
        if idx % 30 == 0:
            print(f"x={x}, x^2={round(x_squared, 6)}, neumann={round(neumann_delta, 6)}, manual={round(manual_nn_delta, 6)}, torch={round(torch_nn_delta, 3)}")
        assert abs(neumann_delta) < 1e-10
        assert abs(manual_nn_delta) < 1e-10
        assert abs(torch_nn_delta) < 1e-10
    print()


if __name__=='__main__':
    runs_square_asserts()
    plot_1()
    plot_2()
