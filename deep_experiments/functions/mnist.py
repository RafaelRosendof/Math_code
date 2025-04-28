import torch 
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch import optim
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    

def train(numero_de_epocas, cnn, loader):
    cnn.train()

    # Treina o modelo
    passos_total = len(loader['train'])    

    for epocas in range(numero_de_epocas):
        for i, (images, labels) in enumerate(loader['train']):
            b_figasX = Variable(images)
            b_figasY = Variable(labels)
            saida = cnn(b_figasX)[0]
            erro = loss_func(saida, b_figasY)

            optimizer.zero_grad()
            erro.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Época [{}/{}], passo [{}/{}], erro: {:.4f}'.format(epocas + 1, numero_de_epocas, i + 1,
                                                                           passos_total, erro.item()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo atual em uso: {device}\n')

train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)
test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
)

print(f'Dados de treinamento: {train_data}\n')
print(train_data.data.size())

print(f'Dados de teste: {test_data}\n')
print(train_data.targets.size())

from torch.utils.data import DataLoader
loaders = {
    'train': torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test': torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
}
print(f'Loaders: {loaders}\n')

cnn = CNN()
print(f'Modelo: {cnn}\n')

loss_func = nn.CrossEntropyLoss()
print(f'Função de perda: {loss_func}\n')

optimizer = optim.Adam(cnn.parameters(), lr=0.01)
print(f'Otimizeador: {optimizer}\n')

num_epochs = 10
train(numero_de_epocas=num_epochs, cnn=cnn, loader=loaders)