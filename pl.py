from torch import optim , utils , Tensor 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST 
from torchvision.transforms import ToTensor 
import lightning as L 
import os 

encoder = nn.Sequential(nn.Linear(28 * 28 , 64), nn.ReLU() , nn.Linear(64,3))
#decode = nn.Sequential(nn.Linear(28 * 28 , 64) nn.Relu() , nn.Linear(64,3))
decode = nn.Sequential(nn.Linear(3 , 64) , nn.ReLU() , nn.Linear(64 , 28 * 28))

class LitEncoder(L.LightningModule):
    def __init__(self , encoder , decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        
    def training_step(self, batch , batch_idx):
        x , _ = batch 
        x = x.view(x.size(0) , -1)
        z = self.encoder(x)
        x2 = self.decoder(z) 
        
        
        loss = nn.functional.mse_loss(x2 , x)
        
        self.log("Perca do treinamento", loss)
        
        return loss 
    
    
    def config_optim(self):
        optimi = optim.Adam(self.parameters() , lr=1e-2)
        return optimi
    
    
autoencoder = LitEncoder(encoder , decode)

data = MNIST(os.getcwd() , download=True , transform=ToTensor())
train_loader = utils.data.DataLoader(data)

trainer = L.Trainer(limit_train_batches=10 , max_epochs=1)

trainer.fit(model = autoencoder , train_dataloaders=train_loader)

