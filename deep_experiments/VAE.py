from tqdm import tqdm
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import lightning as L
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score 
import torchvision.transforms as transforms



#### This gonna be with cifar 10 


class VEAencoder(nn.Module):
    def __init__(self , in_chan , latent_dim):
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chan, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # At this point, we don't know the flattened shape after convs.
        self.flatten_dim = 1024 * 2 * 2  # for input 28x28, 4 downsamplings halve each time
        
        self.mu = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(bs, -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    
    pass 


class VEA(L.LightningModule):
    def __init__(self , in_chan , latent_dim):
        super(VEA , self).__init__()
        
        self.encoder = VEAencoder(in_chan , latent_dim)
        self.decoder = Decoder(in_chan , latent_dim)
        
        self.save_hyperparameters()
        self.validation_step_outputs = []
        
    def forward(self , x):
        #encode = self.encoder(x)
        #decode = self.decoder(encode)
        
        #return decode
        mu , logvar = self.encoder(x)
        
        std = torch.exp(0.5 * logvar) # distrib 
        
        y = self.sample(mu , logvar)
        
        reconstructed = self.decoder(y)
        
        return reconstructed , mu , logvar
    
    def sample(self , mu , std):
        #sample from n(0 , i) 
        #standard normal distribution and scale     
        eps = torch.randn_like(std)
        return mu + eps * std 
    
    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
    
    def on_validation_batch_end(self):
        self.validation_step_outputs=[]
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters() , lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim , mode='min' , factor=0.5 , patience=2
        )
        
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
        
        
    
