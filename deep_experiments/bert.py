import pandas as pd
import re
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


sentiment_map = {
    "Extremely Negative": "Negative",
    "Negative": "Negative",
    "Neutral": "Neutral",
    "Positive": "Positive",
    "Extremely Positive": "Positive"
}


class MyData(Dataset):
    def __init__(self , texto , labels , tokenizer , max_len):
        self.texto = texto
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        texto = str(self.texto[index])
        label = self.labels[index]
        
        encoding = self.tokenizer(
            texto,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        
        
class SentimentalModule(L.LightningDataModule):
    def __init__(self, train_df_path , test_df_path , batch_size=64 , max_len=128, num_workers=4):
        self.train_df_path = train_df_path
        self.test_df_path = test_df_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self._log_hyperparams = False
        
    def prepare_data(self):
        pass 
    
    def setup(self , stage=None):
        train_df = pd.read_csv(self.train_df_path, encoding='latin')
        test_df = pd.read_csv(self.test_df_path ,encoding='latin')

        sentiment_map = {
            "Extremely Negative": "Negative",
            "Negative": "Negative",
            "Neutral": "Neutral",
            "Positive": "Positive",
            "Extremely Positive": "Positive"
        }
        
        train_df["Sentiment"] = train_df["Sentiment"].map(sentiment_map)
        test_df["Sentiment"] = test_df["Sentiment"].map(sentiment_map)
        
        #limpando 
        train_df["clean_text"] = train_df["OriginalTweet"].apply(clean_text)
        test_df["clean_text"] = test_df["OriginalTweet"].apply(clean_text)
        
        #tabelando rotulos
        sentiment_mapping = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        }
        
        train_df["labels"] = train_df["Sentiment"].map(sentiment_mapping)
        test_df["labels"] = test_df["Sentiment"].map(sentiment_mapping)
        
        if stage == 'fit' or stage is None:
            X = train_df["clean_text"]
            Y = train_df["labels"]
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, Y, test_size=0.2, random_state=40, stratify=Y
            )
            
            self.train_dataset = MyData(
                X_train.tolist(),
                y_train.tolist(),
                self.tokenizer,
                self.max_len
            )
            
            self.val_dataset = MyData(
                X_val.tolist(),
                y_val.tolist(),
                self.tokenizer,
                self.max_len
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = MyData(
                test_df["clean_text"].tolist(),
                test_df["labels"].tolist(),
                self.tokenizer,
                self.max_len
            )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )       


class BERTSentimentClassifier(L.LightningModule):
    def __init__(self, n_classes=3, learning_rate=2e-5):
        super().__init__()
        self.n_classes = n_classes
        self.learning_rate = learning_rate

        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained("bert-base-cased")
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        
        # Save hyperparameters
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Pass through classifier
        return self.classifier(pooled_output)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Log training metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Log validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        self.validation_step_outputs.append({'val_loss': loss, 'preds': preds, 'labels': labels})
        
        return {'val_loss': loss, 'preds': preds, 'labels': labels}
    
    def on_validation_epoch_end(self):
        # Aggregate predictions and calculate F1 score
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Calculate F1 score (weighted for imbalanced classes)
        f1 = f1_score(labels_np, preds_np, average='weighted')
        self.log('val_f1', f1, prog_bar=True)
        
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        # Log test metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        self.test_step_outputs.append({'test_loss': loss, 'preds': preds, 'labels': labels})
        
        return {'test_loss': loss, 'preds': preds, 'labels': labels}
    
    def on_test_epoch_end(self):
        # Aggregate predictions and calculate F1 score
        preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Calculate F1 score (weighted for imbalanced classes)
        f1 = f1_score(labels_np, preds_np, average='weighted')
        self.log('test_f1', f1, prog_bar=True)
        
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        # Use AdamW optimizer as recommended for BERT
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler with linear warmup and decay
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
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


def main():
    train_ph = "Corona_NLP_train.csv"
    test_ph = "Corona_NLP_test.csv"
    
    data_module = SentimentalModule(
        train_df_path=train_ph,
        test_df_path=test_ph,
        batch_size=16,
        max_len=128
    )
    
    model = BERTSentimentClassifier(n_classes=3, learning_rate=2e-5)
    
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu", 
        devices=2,
        strategy="fsdp",
        precision="32",  
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val_f1",
                mode="max",
                save_top_k=1,
                filename="{epoch}-{val_f1:.2f}"
            ),
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_f1",
                patience=4,
                mode="max"
            )
        ]
    )
    
    test_trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32"
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    trainer.save_checkpoint("best_model.ckpt")
    
    test_model = BERTSentimentClassifier.load_from_checkpoint("best_model.ckpt")
    
    test_trainer.test(test_model, data_module)
    
def inference(model , tokenizer , text):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        
    return predictions.item()

if __name__ == "__main__":
    main()
    
# Inference example
# Load the tokenizer and model
    model = BERTSentimentClassifier.load_from_checkpoint("best_model.ckpt")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    text = "I love this product! It's amazing." 
    text2 = "Olá meu amigo eu me chamo Rafael e eu gosto de cerejas"
    
    inference(model , tokenizer , text)
    inference(model , tokenizer , text2)