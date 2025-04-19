import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'ME FODI')
print(device)

class simples(nn.Module):
    def __init__(self, input_size , escondida_size , saida_size):
        super(simples , self).__init__()
        self.fc1 = nn.Linear(input_size , escondida_size) # primeira camada entrada de uma 
        self.fc2 = nn.Linear(escondida_size, saida_size) #segunda camada saida da outra

    def forward(self , x):
        x = torch.relu(self.fc1(x)) #Aplicando a função de ativação RELU na saida da primeira camada
        x = self.fc2(x) #saida final 
        return x
#Criando as instâncias do modelo e levando para a placa de video
input_size = 20 #camadas de entrada
escondida_size = 100 #camadas escondidas no meio
saida_size = 2 #camada binária de saida
modelo = simples(input_size, escondida_size , saida_size).to(device)
#carregando as camadas no modelo
#movendo os dados de exemplo e os rótulos para a GPU

X = torch.randn(100,input_size).to(device) # # 100 exemplos, cada um com 10 características
y = torch.randint(0,2,(100,)).to(device)

criterio = nn.CrossEntropyLoss() #função de perda: entropia cruzada 
optimizer = optim.SGD(modelo.parameters(),lr=0.01) # otimizador: SGD com taxa de aprendizado de 0.01 

epochs = 10000 # quantidade de épocas 

for epoch in range(epochs): #função de treino 
    outputs = modelo(X)
    loss = criterio(outputs,y)

    #backwards pass e otimizador
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch + 1) % 10 ==0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
