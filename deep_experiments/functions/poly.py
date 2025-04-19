import torch 
import torch.nn as nn
import numpy as np
import torch.optim as optim

x_treino = np.random.rand(100,1) * 10 - 5
y_treino = 2 * x_treino**4 - 3 * x_treino**3 + 4 * x_treino**2 - 5 * x_treino + 6

x_treino_tensor = torch.FloatTensor(x_treino).cuda()
y_tensor = torch.FloatTensor(y_treino).cuda()

class polinomio(nn.Module):
    def __init__(self):
        super(polinomio , self).__init__()
        self.fc1 = nn.Linear(1,64) #camada de entrada
        self.fc2 = nn.Linear(64,64) #camada oculta
        self.fc3 = nn.Linear(64,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

modelo = polinomio()

device = torch.device("cuda" if torch.cuda.is_available() else "lascou")
modelo.to(device)

for param in modelo.parameters():
    param.data = param.data.cuda()

criterio = nn.MSELoss()
otimizador = optim.AdamW(modelo.parameters(), lr=0.0001)

tempo = 100000

for epoch in range(tempo):
    saidas = modelo(x_treino_tensor)
    perda = criterio(saidas , y_tensor)

    #processo de backwards
    otimizador.zero_grad()
    perda.backward()
    otimizador.step()

    if(epoch+1) % 100 == 0:
        print(f'Época [{epoch+1}/{tempo}] , Com uma perda de {perda.item():.4f}')

x_teste = torch.FloatTensor(np.linspace(-5,5,100).reshape(-1,1)).cuda()

with torch.no_grad():
    predito = modelo(x_teste).cpu()

    
# Plotando os resultados
import matplotlib.pyplot as plt
plt.scatter(x_treino, y_treino, label='Polinômio')
plt.plot(x_teste.cpu().numpy(), predito.numpy(), 'r-', label='Predito')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Tentando aproximar polinômios com torch')
plt.legend()
plt.show()