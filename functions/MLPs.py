#MLP's com camadas lineares para o treino de uma base mnist com redes lineares normais
# E com funções de ativação ReLU
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

class Redes(nn.Module):
    def __init__(self):
        super(Redes,self).__init__()
        self.figas1 = nn.Linear(28 * 28 , 128) # Tamanho da imagem e a quantidade de neuronios
        self.figas2 = nn.Linear(128,64) # Segunda rede, entrada de uma, saida da outra
        self.figas3 = nn.Linear(64,10) #Última camada aqui saem 10 pois são 10 números

    def forward(self , x):
        x = torch.flatten(x,1) # Achatando a imagem de entrada em um vetor
        x = torch.relu(self.figas1(x))#Aplicando a função relu na primeira rede
        x = torch.relu(self.figas2(x))#Aplicando na segunda
        x = self.figas3(x)
        return x



transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5))]) #Transformações para normalizar a imagem
train_dataset = datasets.MNIST(root='./data' , train=True , download=True, transform = transform)
test_dataset = datasets.MNIST(root='./data',train=False , download=True, transform = transform)

treino_data = DataLoader(train_dataset , batch_size = 64 , shuffle=True)
test_loader = DataLoader(test_dataset , batch_size = 64 , shuffle=False)

model = Redes()

#Colocar em gpus
device = torch.device("cuda" if torch.cuda.is_available() else "lascou")
model.to(device)

#Tem mais de uma GPU?
if torch.cuda.device_count() > 1:
   print("Usando", torch.cuda.device_count(),"GPUs!")
   model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()# Função de perda para problemas de classificação
optimizer = optim.Adam(model.parameters() , lr=0.002)#otimizador adam e taxa de aprendizado

#Treinamento do modelo
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    run_loss = 0.0
    for imagem , labels in treino_data:
        optimizer.zero_grad() #zerando os gradientes acumulados dos parâmetros 
        imagem, labels = imagem.to(device), labels.to(device)
        outputs = model(imagem)# Passando as imagens pelo modelo para obter novas previsões 
        loss = criterion(outputs , labels)#calculando o erro das previsões 
        loss.backward()#retropropagando os parâmetros do modelo com base no gradiente
        optimizer.step()#Atualizando os parâmetros do modelo com base nos gradientes
        run_loss += loss.item() * imagem.size(0)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {run_loss/len(treino_data):.4f}")

model.eval()#Definindo o modelo no modo de avaliação(Desligando a acamada de dropout)
correct = 0
total = 0
with torch.no_grad():  # Desativando o cálculo dos gradientes durante a avaliação
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)  # Passando as imagens de teste pelo modelo para obter as previsões
        _, predicted = torch.max(outputs, 1)  # Obtendo as previsões com a maior probabilidade
        total += labels.size(0)  # Incrementando o contador de exemplos de teste
        correct += (predicted == labels).sum().item()  # Incrementando o contador de previsões corretas
accuracy = correct / total  # Calculando a acurácia do modelo
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
