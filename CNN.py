import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torchvision

'''
@author: Rafael
@date: 12/02/24
@brief: Convolutional basic mnist script 
'''

#Usamos um script para treinar uma rede neural com redes convolucionais, lineares e com um maxpooling
#Sempre usando gpus para fazer o serviço 

#Vamos definir aqui a estrutura do Modelo
class CNN(nn.Module):
    #contrução de como que será as redes neurais 
    def __init__(self):
        super(CNN, self).__init__() #pergunte o que é isso e as outras camadas o que são?
        self.r1 = nn.Conv2d(in_channels = 1 , out_channels = 32 , kernel_size=3 , stride = 1, padding=1) #primeira camada 
        self.r2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, stride= 1 ,padding=1) # segunda camada 
        self.pool = nn.MaxPool2d(kernel_size=2 , stride=2 , padding=0)
        self.lin1 = nn.Linear(64 * 7 * 7, 128)
        self.lin2 = nn.Linear(128,10)

#função forward, aplicação em X das redes 
    def forward(self,x):
        x = self.pool(torch.relu(self.r1(x))) #aplicando o pooling e a relu na primeira conv2d
        x = self.pool(torch.relu(self.r2(x)))
        x = x.view(-1, 64 * 7 * 7) # Redimensionando para o formato esperado pelas camadas totalmente conectadas
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        return x
# convertendo as imagens PIL para tensores        e Normalizando as imagens 
transform = transforms.Compose([transforms.ToTensor() , transforms.Normalize((0.5,) , (0.5,))])

#Datasets 
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#loader dos datasets
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

model = CNN()
#GPUS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) #modelo na gpu
#tem mais de uma GPU?
if torch.cuda.device_count() > 1:
    print("Usando",torch.cuda.device_count(),"GPU's!!!!!!!")
    model = nn.DataParallel(model , device_ids=[0,1,2])

'''
net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
output = net(input_var)  # input_var can be on any device, including CPU
'''

criterion = nn.CrossEntropyLoss() #Função de perda para problemas de classificação 
optmizer = optim.Adam(model.parameters() , lr=0.01) # otimizador adam e taxa de aprendizado 

#treinamento do modelo 
num_epochs = 100
for epoca in range(num_epochs):
    model.train()
    delta_perda = 0.0
    for imagem , labels in train_loader:
        optmizer.zero_grad() # zerando os gradientes acumulados dos parâmetros de treinamento 
        imagem , labels = imagem.to(device) , labels.to(device)
        outputs = model(imagem) #passando as imagens pelo modelo para obter novas previsões 
        loss = criterion(outputs , labels) #calculando o erro das previsões 
        loss.backward() # retroporpagando os parâmetros do modelo com base no gradiente
        optmizer.step() #atualizando os parâmetros do modelo com base nos gradientes
        delta_perda += loss.item() * imagem.size(0)
    print(f"Epoch {epoca+1}/{num_epochs}, Loss: {delta_perda/len(train_loader):.4f}")
    
model.eval()#Definindo o modelo no modo de avaliação(Desligando a acamada de dropout)
correto = 0
total = 0
with torch.no_grad(): #desativando o cálculo dos gradientes durante a avaliação 
    for images ,labels in test_loader: #vamos ver se o negócio aprendeu
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images) # passando as imagens para o teste
        _,predicted=torch.max(outputs , 1) #Obtendo as previsões com a maior probabilidade
        total += labels.size(0) #incrementando o contador de testes
        correto += (predicted == labels).sum().item()#Incrementando o contador de previsões 
accuracy = correto / total #acurácia?
print(f"Resultado final, afinal? Quanto que deu mesmo?: {accuracy:.4f}")    


'''
Algoritmo: Pipeline de Treinamento de uma Rede Neural

Entrada: 
    - Conjunto de treinamento (dados de entrada e rótulos)
    - Conjunto de validação (opcional)
    - Modelo de rede neural
    - Função de perda
    - Otimizador
    - Número de épocas
    - Dispositivo de computação (CPU ou GPU)

Saída:
    - Modelo treinado

Passos:
1. Para cada época no número total de épocas:
    1.1. Coloque o modelo em modo de treinamento.
    1.2. Inicialize a perda acumulada para a época como zero.
    1.3. Para cada lote no conjunto de treinamento:
        1.3.1. Zere os gradientes do otimizador.
        1.3.2. Propague os dados de entrada através do modelo para obter as previsões.
        1.3.3. Calcule a perda entre as previsões e os rótulos usando a função de perda.
        1.3.4. Retropropague os gradientes da perda.
        1.3.5. Atualize os pesos do modelo usando o otimizador.
        1.3.6. Adicione a perda do lote à perda acumulada da época.
    1.4. Calcule a perda média da época dividindo a perda acumulada pelo número de lotes.
    1.5. Se houver um conjunto de validação:
        1.5.1. Avalie o modelo no conjunto de validação.
        1.5.2. Registre a métrica de desempenho (por exemplo, acurácia) no conjunto de validação.
    1.6. Imprima a perda média da época e, se aplicável, a métrica de desempenho no conjunto de validação.
2. Após todas as épocas, retorne o modelo treinado.


'''