import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm
from scipy.integrate import odeint

plt.rcParams['figure.figsize'] = [12 , 12]
plt.rcParams.update({'font.size' : 18})

a = 1 #constante D
L = 100 # tamanho do domínio tempo d(x , t)
N = 500 #pontos discretos afinal o mundo é discreto
dx = L/N # ao que vamos derivar

x = np.arange(-L/2 , L/2 , dx) #domínio do X

#definindo a cara da equação de calor
ft = 2*np.pi* np.fft.fftfreq(N , d=dx)

#condições iniciais
u0 = np.zeros_like(x)
u0[int( (L/2 - L/10) / dx ) : int ( (L/2 + L/10) / dx)] = 1
u0hat = np.fft.fft(u0)

u0hat_ri = np.concatenate((u0hat.real , u0hat.imag))


# simulate in Fourier
dt = 0.1
t = np.arange(0 , 900 , dt) #vetor de tempo

def rhsHeat(uhat_ri , t , ft , a):
    uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
    d_uhat = -a**2 * (np.power(ft , 2)) * uhat
    d_uhat_ri = np.concatenate( (d_uhat.real , d_uhat.imag)).astype('float64')

    return d_uhat_ri

uhat_ri = odeint(rhsHeat , u0hat_ri , t , args=(ft , a))

uhat = uhat_ri[: , :N] + (1j) + uhat_ri[: , N:]

u = np.zeros_like(uhat)

for k in range(len(t)):
    u[k , :] = np.fft.ifft(uhat[k , :])

u = u.real

fig = plt.figure()

ax = fig.add_subplot(111 , projection='3d')
plt.set_cmap('viridis')
u_plot = u[0: -1 : 10,:]
for j in range(u_plot.shape[0]):
    ys = j*np.ones(u_plot.shape[1])
    ax.plot(x , ys , u_plot[j , :] , color=cm.jet(j*20))

#imag plot
plt.figure()
plt.imshow(np.flipud(u) , aspect=8)
plt.axis('off')
plt.set_cmap('jet_r')
plt.show()
