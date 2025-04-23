from torch import optim , utils , Tensor 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor 
import lightning as L 
import torch
import os 



cifar10 = torchvision.datasets.CIFAR10(
    root=os.path.join(os.getcwd(), 'data'),
    train=True,
    download=True,
    transform=ToTensor()
)
cifar10_test = torchvision.datasets.CIFAR10(
    root=os.path.join(os.getcwd(), 'data'),
    train=False,
    download=True,
    transform=ToTensor()
)

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.cifar10_train = cifar10
        self.cifar10_test = cifar10_test

    def train_dataloader(self):
        return utils.data.DataLoader(
            self.cifar10_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return utils.data.DataLoader(
            self.cifar10_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    def test_dataloader(self):
        return utils.data.DataLoader(
            self.cifar10_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
        
        
class CIFAR10Model(L.LightningModule):
    def __init__(self, learning_rate: float = 0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss
    def test_step(self, batch: tuple, batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
if __name__ == "__main__":
    # Initialize the data module
    cifar10_data = CIFAR10DataModule(batch_size=256, num_workers=4)
    cifar10_data.setup(stage='fit')

    # Initialize the model
    model = CIFAR10Model(learning_rate=0.001)

    # Initialize the trainer
    #trainer = L.Trainer(max_epochs=5, gpus=1)

    trainer = L.Trainer(limit_train_batches=10 , 
                    max_epochs=120,
                    overfit_batches=1,
                    min_epochs=5,
                    precision=32,
                    accelerator="gpu",
                    )

#strategy = fsdp or deepspeed_stage_2" 
#devices= 4 or other one 


    # Train the model
    trainer.fit(model, cifar10_data)