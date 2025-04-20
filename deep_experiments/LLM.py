
from posix import listdir
import pandas as pd
import re,os
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoTokenizer, AutoModel, AdamW
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pyarrow.parquet as pq
import pandas as pd


'''
dir_path = "/home/rafael/Math_code/deep_experiments/TucanoBR___wikipedia-pt"

df = pd.concat([
    pd.read_parquet(os.path.join(dir_path, f))
    for f in os.listdir(dir_path)
    if f.endswith(".parquet")
])
print(df.head())
    #Cria somente 1 pd dataframe

df.to_csv("data.csv",index=False)
'''



#dataset = load_dataset("TucanoBR/wikipedia-PT")

#print(dataset)


### Define my dataset
'''
For this dataset, we hava a basic dataset that contain only text, and  this text is based by a wikipedia text and separed by ',' is a csv file
We only gonna pass all the data give by a limit of carachters
'''


### Create my dataset class
class MyDataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer,
        max_len
    ): # we can pass the batchsize here?
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length = self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}

        return item

### Create my dataModule

class MyDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_df_path,
        batch_size,
        max_len,
        num_workers,
        model_tokenizer
    ):
        super().__init__()
        self.train_df_path = train_df_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        def prepare_data(self):
            pass #verificar melhor esse depois

        def setup(self , stage=None):

            #train_df = pd.read_csv(self.train_df_path , encoding='utf-8')

            #X_train = train_df['text']
            #Don't need to use test

            with open(self.train_df_path , 'r' , encoding='utf-8') as f:
                texts = f.read().split('Agrotis')
                texts = ['Agrotis' + text if i > 0 else text for i , text in enumerate(texts)]
                texts = [text for text in texts if text.strip()]



            if stage == 'fit' or stage is None:
                self.train_dataset = MyDataset(
                    texts,
                    self.tokenizer,
                    self.max_len
                )


            #again do not need a validation or test

        def train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )


### Create my model class

class LLM_model(L.LightningModule):
    def __init__(
        self,
        model_name,
        lr = 1e-7,
    ):

        super().__init__()
        self.lr = lr
        self.model_name = model_name
        self.LLM = AutoModel.from_pretrained(model_name)
        self.save_hyperparameters()

    def forward(self , input_ids , attention_mask , labels=None):
        #get llm outputs
        out = self.LLM(
            input_ids=input_ids ,
            attention_mask=attention_mask,
            labels=labels)

        return out

    def training_step(self , batch , batch_idx):
        outps = self(
           input_ids=batch['input_ids'],
           attention_mask=batch['attention_mask'],
           labels=batch['labels']
       )

        loss = outps.loss
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        # Use AdamW optimizer as recommended for LLM?
        optimizer = AdamW(self.parameters(), lr=self.lr)

        # Learning rate scheduler with linear warmup and decay
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1  # 10% of training for warmup
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

### Create my Trainer module

### Trainer and loggs function

### Main function
